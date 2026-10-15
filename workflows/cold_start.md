
# 方案设计与冷启动工作流

本工作流只负责 `v_0`。

每个小节标题中的步骤 ID 对应 `status.yaml` 的 `current_step`，格式为 `cold_start:step_XX`。

**模型约定**：本工作流一次只针对一个目标模型执行（`task.yaml` 只启用一个候选时可缺省 `--model`，多个启用时所有脚本命令需加 `--model <key>`）。该模型的全部产物（standard 到 scan）都在 `<model>/v_0/` 下。多个模型 = 多次独立执行本工作流，互不干扰。

**laya 候选（`type: laya`）差异**：完整约定见 `SKILL.md` §17。步骤 1–7、12–14 照常执行；步骤 8（合成 Prompt）与 9（合成数据）跳过；步骤 10 只构建 validation + regression_eval（不产出 train）；步骤 11 不做训练与超参搜索，只对 validation 上的 `laya_prob` 做 threshold 搜索（网格与 objective 读 `task.yaml` `training.threshold_search` / `acceptance.precision_min`），产物写入 `training/threshold_search.csv` 与 `training/best_threshold.yaml`。question 配置每版本冻结为 `standard/laya_question.yaml`，state 按 `context.bert_turns` 用 `src/text_utils` 统一拼接。

目标是从标签定义和少量人工样本出发，建立第一版：

```text
标签标准
→ Regex
→ LLM 标注体系
→ 训练数据
→ BERT 模型
→ threshold
→ 全量扫描结果
```

---

## 1. 初始化 v_0（step_01_init）

创建（该模型的流水线版本目录，承载全部产物）：

```text
<model>/v_0/
```

并初始化：

```text
<model>/v_0/manifest.yaml
<model>/v_0/status.yaml
```

初始状态：

```yaml
status: initializing
```

检查以下输入是否存在：

```text
task.yaml
data_schema.yaml
用户提供的人工标注数据
线上全量数据
候选 BERT 模型
```

如果缺少必要输入，停止执行并记录缺失项。

---

## 2. 标签标准设计（step_02_standard_design）

根据：

* 标签定义；
* 人工标注样本；
* 当前业务场景；

分析正例内部可以稳定区分的细分子场景。

生成：

```text
<model>/v_0/standard/
├── label_definition.md
├── scenes.yaml
└── boundary_cases.jsonl
```

`scenes.yaml` 中每个子场景至少包含：

```yaml
scene_id: scene_001
scene_name: ""
definition: ""
positive_rule: ""
negative_rule: ""
```

每条正例必须能够归属到一个 `scene_id`。

`boundary_cases.jsonl` 同时提供：

* 明确正例；
* 明确负例；
* 边界正例；
* 易混淆负例。

完成初稿后：

```yaml
status: waiting_standard_review
```

暂停，等待人工确认。

未经人工确认，不进入后续步骤。

---

## 3. Regex 初筛设计（step_03_regex）

人工确认标签标准后，根据：

```text
label_definition.md
scenes.yaml
boundary_cases.jsonl
```

生成：

```text
<model>/v_0/standard/regex/
├── regex_config.yaml
├── regex_cases.jsonl
└── regex_report.yaml
```

Regex 使用 `task.yaml` 中人工指定的上下文轮数。

禁止自行修改轮数。

执行：

```text
scripts/run_regex_prefilter.py
```

对线上全量数据进行初筛。

输出：

```text
<model>/v_0/prefilter/
├── recalled.xlsx
├── recalled.jsonl
├── rejected_sample.xlsx
├── rejected_sample.jsonl
└── prefilter_report.yaml
```

`rejected_sample` 用于抽样检查 Regex 漏召回。

每条初筛产物携带 `data_date`（来自线上数据按日期命名的文件，如 `2026-05-01_recruitment_full_cc2.json`）。线上数据随日期增长，处理新增日期时使用增量模式，避免重扫历史数据：

```text
scripts/run_regex_prefilter.py --version v_0 --model <model> \
  --since 2026-06-01 --append
```

`--since/--until` 按文件名日期过滤输入文件；`--append` 将本次召回按 sample_id 去重后合并进已有 `recalled.jsonl`（重复时以本次结果覆盖）。

---

## 4. LLM 打标 Prompt（step_04_llm_prompt）

基于确认后的标签标准生成：

```text
<model>/v_0/standard/llm_label_prompt.md
```

Prompt 必须要求 LLM 至少输出：

```text
label
scene_id
reason
```

其中：

* `label=positive` 时必须输出合法 `scene_id`；
* `label=negative` 时 `scene_id` 为空。

LLM 标注使用人工给定的完整对话或上下文轮数。

---

## 5. 冷启动采样与 LLM 标注（step_05_coldstart_labeling）

从 Regex 初筛结果中采样 2000 条（数量读 `task.yaml` `cold_start_labeling.sample_size`）。

执行（采样快照与打标一步完成，`--save-sample` 落盘冷启动样本文件）：

```text
scripts/label_llm.py --version v_0 \
  --input <model>/v_0/prefilter/recalled.jsonl \
  --llm LongCat-Flash-Chat \
  --sample-size 2000 --save-sample
```

保存：

```text
<model>/v_0/labeling/coldstart_sample_2000.xlsx
<model>/v_0/labeling/coldstart_sample_2000.jsonl
```

调用：

```text
LongCat-Flash-Chat
```

按照：

```text
<model>/v_0/standard/llm_label_prompt.md
```

完成打标。

保存原始输出：

```text
<model>/v_0/labeling/llm/raw/longcat_flash.jsonl
```

保存标准化结果：

```text
<model>/v_0/labeling/llm/longcat_flash.xlsx
<model>/v_0/labeling/llm/longcat_flash.jsonl
```

---

## 6. 生成人工复核数据（step_06_human_review）

从 LLM 标注结果中选择：

```text
全部正例
+
max(2 × 正例数量, 100) 条负例
```

执行（复核表导出与人工填写列自动生成）：

```text
scripts/export_review_input.py --version v_0 --mode cold_start \
  --input <model>/v_0/labeling/llm/longcat_flash.jsonl \
  --output <model>/v_0/labeling/human/review_input
```

生成：

```text
<model>/v_0/labeling/human/review_input.xlsx
```

同时保存对应 JSONL 快照。

更新：

```yaml
status: waiting_human_review
```

暂停执行。

等待人工完成：

```text
<model>/v_0/labeling/human/review_done.xlsx
```

检测到完成文件后调用：

```text
scripts/import_human_review.py
```

生成：

```text
review_result.xlsx
review_result.jsonl
```

人工结果作为该批样本最终标签。

同时从 `review_result` 中构建初版 Gold 数据：

```text
scripts/build_gold.py --version v_0 --input <model>/v_0/labeling/human/review_result.jsonl
```

* 只使用 `human_verified=true` 的真实样本；
* 默认全量采纳（`--mode all`，SKILL.md §8）：本轮全部人工核验样本合入 gold，按 `sample_id` 去重；
* 需要控制 gold 规模时可用 `--mode sample`（正例按 `scene_id` 分层抽样、易混淆负例优先）；
* 保存到根目录 `data/gold/`（gold_positive / gold_negative 的 xlsx 与 jsonl）；
* 进入 gold 的样本同时从 train / validation 候选池中剔除（`build_dataset.py` 自动执行），保证 gold 与训练数据完全隔离。

Gold 的后续维护（SKILL.md §8）：

* 不进入 train / validation，不参与任何调优；
* 每次迭代人工复核完成后，新增人工核验样本继续全量追加（iteration.md 步骤 4）；
* 需求变更需要修正历史 label 时，通过 `build_gold.py --update-labels --reason` 改标并在 `label_history` 留审计，禁止直接手改 gold 文件。

---

## 7. 数据校准（step_07_calibration）

根据人工复核结果进行：

* FP 分析；
* FN 分析；
* 子场景覆盖分析；
* 边界 case 分析；
* LLM 误判分析。

生成：

```text
<model>/v_0/labeling/calibration/
├── error_analysis.md
├── positive_ratio.yaml
└── calibration_report.md
```

根据分析结果更新：

```text
standard/scenes.yaml
standard/boundary_cases.jsonl
standard/llm_label_prompt.md
standard/regex/
```

如果标签边界发生重大变化，再次进入：

```text
waiting_standard_review
```

由人工确认后继续。

---

## 8. 生成合成 Prompt（step_08_synthesis_prompts）

为每个细分子场景创建：

```text
<model>/v_0/standard/synthesis_prompts/
├── scene_001.md
├── scene_002.md
└── ...
```

Prompt 必须依据最新：

```text
scene definition
positive cases
negative cases
boundary cases
```

生成与真实线上表达尽可能接近的正例。

---

## 9. 判断是否需要合成数据（step_09_synthesis）

满足以下任一条件时进行数据合成：

```text
总体正例率 < 20%
```

或：

```text
任一子场景已有真实/模型正例数量 < 20
```

如果需要合成，对每个需要补充的子场景分别调用（模型列表以 `task.yaml` 的 `synthesis.models` 为准，默认三个）：

```text
LongCat-2.0
LongCat-Flash-Chat
deepseek-v4-flash
```

默认每个模型、每个目标子场景生成 100 条正例。

执行方式（每个模型分别运行）：

```text
scripts/generate_synthetic.py --version v_0 --mode scene \
  --scenes scene_001 scene_002 --llm LongCat-2.0
```

结果分别保存至：

```text
<model>/v_0/synthesis/<model_name>/
```

合并结果保存：

```text
merged_synthetic.xlsx
merged_synthetic.jsonl
```

每条合成数据必须记录：

```text
scene_id
data_source=synthetic
generator_model
synthesis_prompt_version
version=v_0
```

---

## 10. 构建数据集（step_10_build_dataset）

调用：

```text
scripts/build_dataset.py
```

构建：

```text
<model>/v_0/dataset/
├── train.xlsx
├── train.jsonl
├── validation.xlsx
├── validation.jsonl
├── regression_eval.xlsx
├── regression_eval.jsonl
└── statistics.yaml
```

基本原则：

```text
模型标注数据 + 合成数据
→ train
```

用于模型选择和 threshold 调优的数据：

```text
→ validation
```

**validation 只允许来自人工复核过的样本**（`human_verified=true`）——threshold 与超参若在 LLM 标注数据上调优，等于拟合 LLM 的偏差。`build_dataset.py` 已按此实现（validation 只从 `--human` 输入切分），禁止把 LLM 标注或合成数据切进 validation。

人工确认但明确要求只用于历史错误评估的数据：

```text
→ regression_eval
```

根目录：

```text
data/gold/
```

始终独立，不加入训练。

数据拆分时避免同一个 conversation 同时进入 train 与 validation。

---

## 11. BERT 模型训练（step_11_train_bert）

对目标模型执行超参网格搜索（网格读 `task.yaml` `training.hparam_search`，每组内自动完成 `train_bert.py` 训练 + `tune_threshold.py` 阈值搜索；训练数据与产物都在该模型版本目录下）：

```text
scripts/hparam_search.py --version v_0 --model <model> \
  --train <model>/v_0/dataset/train.jsonl --validation <model>/v_0/dataset/validation.jsonl
```

多卡并行（总任务 = 网格组数，按分片编号分配）：

```text
CUDA_VISIBLE_DEVICES=0 scripts/hparam_search.py --version v_0 --model <model> ... --config-idx 0 --num-shards 4
CUDA_VISIBLE_DEVICES=1 scripts/hparam_search.py --version v_0 --model <model> ... --config-idx 1 --num-shards 4
...
```

候选模型（task.yaml 可启用 1 个或多个，各模型独立流水线，本步骤只训练目标模型）：

```text
h256
h288
macbert
```

输出到：

```text
<model>/v_0/training/
```

至少保存：

```text
cfg<i>/
tuning_trials.csv
best_hparams.yaml
threshold_search.csv（cfg<i>/ 内）
best_threshold.yaml
validation_predictions.xlsx（cfg<i>/ 内）
best_model/（cfg<i>/ 内）
```

BERT 训练和推理必须使用完全相同的上下文拼接逻辑。

上下文轮数严格读取 `task.yaml`。

---

## 12. 模型评估（step_12_evaluate）

调用（对目标模型的最优配置统一评估；无跨模型比较——各模型独立流水线）：

```text
scripts/select_model.py --version v_0 --model <model>
```

统一评估三个数据集：validation（调优）、regression_eval、**gold（验收指标以 gold 为准——threshold 用 validation 调，若 acceptance 也在 validation 上算就是自己调自己）**。

分别统计：

```text
precision
recall
f1
各 scene recall
FP 数量
FN 数量
```

输出（全部在该模型版本目录下）：

```text
<model>/v_0/metrics/
```

包括 gold 上的 FP/FN 明细（`<model>_fp.jsonl` / `<model>_fn.jsonl`，供归因分析）。

评估检查（`select_model.py` 自动执行）：

* gold / validation 主指标（`task.yaml` `model_selection.primary_metric`）；
* regression_eval 不退化（require_regression_no_degradation）；
* 各细分子场景表现。

最终配置写入：

```text
<model>/v_0/metrics/selected_model.yaml
```

验收判断（以 gold 为准）调用：

```text
scripts/evaluate.py --model-dir <selected best_model> --threshold <selected threshold> \
  --dataset gold=<gold_positive.jsonl>,<gold_negative.jsonl> \
  --dataset regression_eval=<model>/v_0/dataset/regression_eval.jsonl \
  --output-dir <model>/v_0/metrics --acceptance
```

记录到：

```text
<model>/v_0/metrics/acceptance.yaml
```

---

## 13. 全量扫描（step_13_full_scan）

使用目标模型和对应 threshold 调用（Regex 配置读 `<model>/v_0/standard/regex/`，输出写入该模型版本目录）：

```text
scripts/full_scan.py --version v_0 --model <model> \
  --model-dir <selected best_model> --threshold <selected threshold>
```

扫描 Regex 初筛后的线上样本。

结果存入：

```text
<model>/v_0/scan/
```

至少生成：

```text
selected_model_predictions.xlsx
selected_model_predictions.jsonl
union_positive.xlsx
union_positive.jsonl
scan_report.yaml
```

---

## 14. 输出 v_0 报告（step_14_report）

生成（该模型的冷启动报告）：

```text
<model>/v_0/report.md
```

报告至少包含：

```text
标签定义
子场景列表
Regex 说明
冷启动数据规模
人工复核规模
总体正例率
各子场景样本数量
合成数据规模
候选模型
最佳超参数
最佳 threshold
Precision / Recall / F1
各子场景指标
全量扫描规模
已知问题
下一步迭代建议
```

完成后（该模型 `v_0` 的 status 置为 completed）：

```yaml
status: completed
```

后续工作进入 `workflows/iteration.md`。
