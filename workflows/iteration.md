# 迭代循环工作流

本工作流适用于：

```text
v_1
v_2
...
v_k
```

每一轮都以上一轮完成版本为基础进行迭代。

每个小节标题中的步骤 ID 对应 `status.yaml` 的 `current_step`，格式为 `iteration:step_XX`。

**模型约定**：本工作流一次只针对一个目标模型执行（`task.yaml` 只启用一个候选时可缺省 `--model`，多个启用时所有脚本命令需加 `--model <key>`）。该模型的全部产物都在 `<model>/v_{k+1}/` 下；上一轮产物在同模型 `<model>/v_k/` 下。多个模型 = 多条独立迭代链，版本号各自计数，互不干扰。

**laya 候选（`type: laya`）差异**：完整约定见 `SKILL.md` §17。步骤 1–7、11–15 照常执行（含 Regex 漏召回检查、验收与报告）；步骤 8（hard case 合成）、9（训练集回流）、10（重训）跳过——无训练集可消费。FP/FN 的修复手段限定为：在 `v_{k+1}/standard/laya_question.yaml` 中修改 `instructions` / `criteria`、重搜 threshold、更新 Regex；真实 FP/FN 照常回流 `regression_eval`。步骤 11 评估与步骤 13 扫描使用 `metrics/selected_model.yaml` 记录的 checkpoint + question + threshold 三元组，按 `laya_prob` 判定。

核心循环：

```text
模型扫描
→ 样本采样
→ 多 LLM 打标
→ 人工复核
→ FP/FN 归因
→ 标准更新
→ hard case 合成
→ 数据集回流
→ 重新训练
→ 重新扫描
→ 指标判断
```

---

## 1. 创建新版本（step_01_new_version）

查找当前最大已完成版本：

```text
v_k
```

创建（该模型的流水线版本目录，承载全部产物）：

```text
<model>/v_{k+1}/
```

在新版本中复制上一轮：

```text
<model>/v_k/standard/
```

作为当前轮初始标签标准（复制到 `<model>/v_{k+1}/standard/`）。

禁止直接修改上一版本。

在：

```text
<model>/v_{k+1}/manifest.yaml
```

记录：

```yaml
version: v_{k+1}
parent_version: v_k
```

初始化：

```yaml
status: initialized
```

---

## 2. 迭代样本采样（step_02_sampling）

以上一轮该模型的 BERT 模型和 threshold 为基础。

输入为本模型上一轮的扫描结果。若上一轮完成后线上数据有增长，先按 SKILL.md §9「数据增长时的版本内刷新」在 `v_k` 上做增量初筛 + 增量重扫（`--since --append`），再开始本轮迭代，保证采样覆盖最新数据。

调用（输出到本轮版本目录）：

```text
scripts/sample_iteration_data.py --version v_{k+1} --model <model> \
  --input <model>/v_k/scan/selected_model_predictions.jsonl
```

对当前 Regex 初筛后的线上数据进行预测并采样。

采样范围：

```text
BERT 预测的全部正例
+
3 × 正例数量的负例
```

负例采样优先考虑：

* threshold 附近样本；
* 历史高频 FP 场景；
* 历史高频 FN 场景；
* 模型低置信样本。

输出：

```text
labeling/candidate_pool.xlsx
labeling/candidate_pool.jsonl
labeling/positive_pool.xlsx
labeling/positive_pool.jsonl
labeling/negative_sample.xlsx
labeling/negative_sample.jsonl
```

---

## 3. 多 LLM 打标（step_03_llm_labeling）

分别调用：

```text
LongCat-2.0
LongCat-Flash-Chat
deepseek-v4-flash
```

使用当前版本：

```text
<model>/v_{k+1}/standard/llm_label_prompt.md
```

每个模型分别执行（断点续跑，可并行）：

```text
scripts/label_llm.py --version v_{k+1} \
  --input <model>/v_{k+1}/labeling/candidate_pool.jsonl --llm <模型名>
```

完成独立打标。

原始返回保存至：

```text
labeling/llm/raw/
```

标准化结果保存：

```text
longcat_2.xlsx/jsonl
longcat_flash.xlsx/jsonl
deepseek_v4_flash.xlsx/jsonl
```

然后调用：

```text
scripts/llm_vote.py
```

生成：

```text
vote_merged.xlsx
vote_merged.jsonl
```

必须保留每个模型独立结果，不允许只保存最终 vote。

**一致样本的后续处理**：vote 一致（`vote_label` 为 positive/negative）的样本不进入本轮人工复核，在步骤 9 由 `build_dataset.py --vote` 自动采纳为最终 label（`annotation_source=llm`）；同时通过步骤 4 的一致抽检估计 vote 自动采纳的错误率。

---

## 4. 人工复核（step_04_human_review）

生成复核表（分歧/失败样本 + 一致样本抽检，`--agree-sample-ratio` 默认 5%，用于估计 vote 采纳错误率；`--bert-pred` 附加 BERT 预测列）：

```text
scripts/export_review_input.py --version v_{k+1} --mode disagreement \
  --input <model>/v_{k+1}/labeling/llm/vote_merged.jsonl \
  --output <model>/v_{k+1}/labeling/human/disagreement_review_input
```

人工复核材料至少应包含：

```text
完整上下文
当前 Turn
BERT prediction
BERT score
三个 LLM 的 label
三个 LLM 的 scene_id
vote result
```

更新：

```yaml
status: waiting_human_review
```

暂停程序。

等待：

```text
disagreement_review_done.xlsx
```

完成后调用：

```text
scripts/import_human_review.py
```

生成：

```text
review_result.xlsx
review_result.jsonl
```

人工标签作为最终真实标签。

随后把本轮新增人工核验样本**全量合入全局 Gold**（SKILL.md §8；gold 为全局资产，按 `sample_id` 去重，已在 gold 中的样本自动跳过）：

```text
scripts/build_gold.py --version v_{k+1} --model <model> \
  --input <model>/v_{k+1}/labeling/human/review_result.jsonl
```

* 只合入 `human_verified=true` 的真实样本；
* 输入样本 label 与 gold 现值冲突时会被跳过并告警——若属于需求变更，改走步骤 6 的 `--update-labels` 路径；
* gold 不属于任何版本，此处只在 status.yaml 记 event，不登记版本 artifacts。

---

## 5. FP/FN 归因分析（step_05_error_analysis）

以人工复核结果为基准，计算 BERT FP/FN（`evaluate.py` 产出 fp/fn 明细），再调用 LLM 辅助归因生成 error_type 初稿（最终以人工复核的 `human_error_type` 为准）：

```text
scripts/classify_errors.py --version v_{k+1} \
  --input <model>/v_{k+1}/labeling/analysis/fp.jsonl <model>/v_{k+1}/labeling/analysis/fn.jsonl \
  --llm LongCat-Flash-Chat
```

输出：

```text
labeling/analysis/
├── fp.xlsx
├── fp.jsonl
├── fn.xlsx
├── fn.jsonl
├── fp_classified.jsonl / fn_classified.jsonl（LLM 归因初稿）
├── disagreement.xlsx
├── disagreement.jsonl
├── error_analysis.md
└── scene_change_report.md
```

FN 默认归因类型：

```text
new_scene
boundary_definition
insufficient_context
bert_capacity
regex_miss
other
```

FP 默认归因类型：

```text
keyword_false_trigger
negation
quoted_content
context_reversal
boundary_definition
other
```

可以新增错误类型，但不得删除历史类型而导致历史结果不可解释。

---

## 6. 标签标准更新（step_06_standard_update）

根据人工结果判断是否需要：

* 新增子场景；
* 修改子场景定义；
* 修改边界 case；
* 修改 LLM Prompt；
* 修改合成 Prompt；
* 修改 Regex；
* 修改 question 定义（laya/kev 等 question 型候选的 instructions/criteria/label 表面形式，含训练与打分脚本中的常量；须保持训练与推理同一份定义）。

question 型候选的标准更新须按 SKILL.md §18 同步 `<model>/question_registry.yaml`（只追加，不回写历史版本；沿用未改也要登记一条 `change_vs_prev: 无` 的记录）。

所有修改只发生在当前版本：

```text
<model>/v_{k+1}/standard/
```

中，禁止回写 `<model>/v_k/standard/`。

如果出现新子场景，必须同步更新：

```text
scenes.yaml
boundary_cases.jsonl
llm_label_prompt.md
synthesis_prompts/
regex/
```

避免只修改某一部分导致不同标准之间不一致。

Regex 更新后必须重新运行：

```text
regex_cases.jsonl
```

进行历史回归测试。

如果标签边界发生重大变化，进入：

```yaml
status: waiting_standard_review
```

等待人工确认后继续。

**需求变更改标**：如果标签边界/业务需求变更导致历史 Gold 样本 label 需要修正（例如某类话术从 positive 改判 negative），在人工确认新标准后，用需求变更重标结果对全局 Gold 改标（SKILL.md §8）：

```text
scripts/build_gold.py --version v_{k+1} --model <model> \
  --input <需求变更人工重标后的 review_result.jsonl> \
  --update-labels --reason "<变更说明，如：健康证话术一律判负例>"
```

* 只允许通过该命令改标，禁止直接手改 `data/gold/`；
* 改标样本在 `label_history` 保留原 label、原版本、审核人、变更版本与原因，并自动在 gold_positive / gold_negative 之间迁移；
* 改标后本轮步骤 11 评估与步骤 14 验收以新 Gold 为准（与上一版本 acceptance 的对比基线随之变化）；
* 受需求变更影响的训练/validation 样本同理只在新版本数据集中生效（禁止回写历史版本 dataset）。

---

## 7. 真实 FP/FN 回流（step_07_regression_reflow）

人工确认的真实 FP/FN：

```text
禁止直接加入 train
```

统一加入：

```text
dataset/regression_eval_delta.xlsx
dataset/regression_eval_delta.jsonl
```

再与上一轮 `regression_eval` 合并生成：

```text
dataset/regression_eval.xlsx
dataset/regression_eval.jsonl
```

所有真实 hard case 必须保留：

```text
error_type
scene_id
source_version
human_verified=true
```

---

## 8. Hard Case 合成（step_08_hardcase_synthesis）

针对当前轮确认的 FP/FN 调用（模型列表以 `task.yaml` 的 `synthesis.models` 为准，默认三个）：

```text
LongCat-2.0
LongCat-Flash-Chat
deepseek-v4-flash
```

执行方式（分层合成，倍数读 `task.yaml` `synthesis.iteration.hardcase_multiplier`）：

```text
scripts/generate_synthetic.py --version v_{k+1} --mode hardcase \
  --hardcase <model>/v_{k+1}/labeling/analysis/fp_classified.jsonl \
             <model>/v_{k+1}/labeling/analysis/fn_classified.jsonl \
  --llm <模型名>
```

生成 hard case 相似训练数据。

默认合成数量：

```text
人工确认 FP/FN 数量 × 3
```

应尽量按照：

```text
scene_id
error_type
```

进行分层合成，而不是仅复制关键词模式。

合成结果保存：

```text
synthesis/hardcase_synthetic.xlsx
synthesis/hardcase_synthetic.jsonl
```

并记录：

```text
source_hardcase_id
scene_id
error_type
generator_model
synthesis_prompt_version
data_source=synthetic
```

---

## 9. 构建当前轮训练集（step_09_build_dataset）

当前轮训练集：

```text
上一轮完整 train
+
当前轮 hardcase synthetic
+
vote 一致样本（build_dataset.py --vote 自动采纳）
+
其他本轮允许加入的合成/LLM 数据
```

执行（`--vote` 采纳一致样本，分歧样本须已在 `--human` 中复核）：

```text
scripts/build_dataset.py --version v_{k+1} \
  --human <model>/v_{k+1}/labeling/human/review_result.jsonl \
  --vote <model>/v_{k+1}/labeling/llm/vote_merged.jsonl \
  --synthetic <model>/v_{k+1}/synthesis/hardcase_synthetic.jsonl \
  --prev-version v_k \
  --regression-delta <model>/v_{k+1}/labeling/analysis/fp_classified.jsonl \
                     <model>/v_{k+1}/labeling/analysis/fn_classified.jsonl
```

形成：

```text
dataset/train_delta.xlsx
dataset/train_delta.jsonl
dataset/train.xlsx
dataset/train.jsonl
```

当前轮真实 FP/FN 不加入 train。

同时维护：

```text
validation.xlsx/jsonl
regression_eval.xlsx/jsonl
```

并生成：

```text
statistics.yaml
```

统计至少包括：

```text
总样本量
正负例数量
各 scene 数量
真实/合成数据比例
各 annotation_source 数量
本轮 train delta 数量
本轮 regression delta 数量
```

---

## 10. 重新训练（step_10_retrain）

对目标模型重新进行（`hparam_search.py` 统一编排，含每组训练 + threshold 搜索；训练数据与产物都在该模型版本目录下）：

```text
scripts/hparam_search.py --version v_{k+1} --model <model> \
  --train <model>/v_{k+1}/dataset/train.jsonl \
  --validation <model>/v_{k+1}/dataset/validation.jsonl
```

超参数搜索
模型训练
threshold 搜索
validation 评估
regression_eval 评估

输出到：

```text
<model>/v_{k+1}/training/
```

保存：

```text
manifest.yaml
tuning_trials.csv
best_hparams.yaml
threshold_search.csv
best_threshold.yaml
validation_predictions.xlsx
regression_predictions.xlsx
best_model/
```

---

## 11. 模型评估（step_11_evaluate）

统一评估（validation + regression_eval + gold；`--prev-acceptance` 用于与上一版本比较）：

```text
scripts/select_model.py --version v_{k+1} --model <model> \
  --prev-acceptance <model>/v_k/metrics/acceptance.yaml
```

生成（全部在该模型版本目录下）：

```text
<model>/v_{k+1}/metrics/model_metrics.csv
<model>/v_{k+1}/metrics/scene_metrics.csv
<model>/v_{k+1}/metrics/regression_metrics.yaml
<model>/v_{k+1}/metrics/selected_model.yaml
<model>/v_{k+1}/metrics/error_analysis.md
```

gold 上的 FP/FN 明细（`<model>_fp/fn.jsonl`）也在该目录下，供归因分析。

至少比较：

```text
当前版本 vs 上一版本
```

检查：

* overall precision；
* overall recall；
* F1；
* 各 scene recall；
* 历史 regression case 是否退化；
* 新增 hard case 是否得到改善。

---

## 12. Regex 漏召回检查（step_12_regex_miss_check）

每轮需要从 Regex 未召回的数据中进行一定规模抽样。

保存：

```text
prefilter/rejected_sample.xlsx
prefilter/rejected_sample.jsonl
```

通过 LLM 或人工检查是否存在真实正例。

若发现：

```text
regex_miss
```

必须进入当前轮：

```text
error_analysis
→ standard/regex 更新
→ Regex regression test
```

否则无法正确评估完整 Pipeline recall。

---

## 13. 全量扫描（step_13_full_scan）

使用当前轮该模型的模型和 threshold。

使用当前轮 Regex（`<model>/v_{k+1}/standard/regex/`）对线上数据初筛，再使用 BERT 精筛：

```text
scripts/full_scan.py --version v_{k+1} --model <model> \
  --model-dir <selected best_model> --threshold <selected threshold>
```

保存（该模型版本目录）：

```text
<model>/v_{k+1}/scan/
```

生成：

```text
selected_model_predictions.xlsx
selected_model_predictions.jsonl
union_positive.xlsx
union_positive.jsonl
scan_report.yaml
```

本轮精筛结果将成为下一轮：

```text
sample_iteration_data.py --input <model>/v_{k+1}/scan/selected_model_predictions.jsonl
```

的输入。

---

## 14. 是否继续迭代（step_14_acceptance）

从 `task.yaml` 读取接受标准。

默认：

```text
precision >= 0.90
recall >= 0.80
```

同时检查 regression_eval 是否出现明显退化。

验收评估（以 gold 为准，带上一版本对照）：

```text
scripts/evaluate.py --model-dir <selected best_model> --threshold <selected threshold> \
  --dataset gold=<gold_positive.jsonl>,<gold_negative.jsonl> \
  --dataset regression_eval=<model>/v_{k+1}/dataset/regression_eval.jsonl \
  --output-dir <model>/v_{k+1}/metrics --acceptance \
  --prev-acceptance <model>/v_k/metrics/acceptance.yaml
```

结果写入：

```text
<model>/v_{k+1}/metrics/acceptance.yaml
```

例如：

```yaml
precision_target: 0.90
recall_target: 0.80

precision_pass: true
recall_pass: false
regression_pass: true

improvement: 0.003
improvement_pass: false

accepted: false
```

**停止保护**（避免无限迭代）：

1. `max_iterations`：迭代版本数（v_1 起）达到 `task.yaml` `acceptance.max_iterations`（默认 5）即停止，输出当前最优结果与未达标项说明；
2. `min_improvement`：与上一版本 gold F1 的提升低于 `acceptance.min_improvement`（默认 0.005）即判定无改善，停止迭代并在报告中说明（数据/标准已收敛，继续迭代收益有限，建议人工介入分析）。

如果：

```text
accepted: false
且未触发停止保护
```

则：

```text
当前版本完成
→ 创建下一版本
→ 从步骤 1 重新开始
```

如果：

```text
accepted: true
```

或触发停止保护，则停止自动迭代并输出最终报告。

---

## 15. 每轮报告（step_15_report）

每轮生成（该模型的迭代报告）：

```text
<model>/v_{k+1}/report.md
```

至少包含：

```text
版本号
父版本
本轮 Regex 变化
新增/修改子场景
人工复核规模
本轮 Gold 追加 / 改标数量（含需求变更说明）
FP/FN 数量
FP/FN 错误类型分布
新增 regression case 数量
合成数据数量
训练集变化
最佳模型
最佳 threshold
Precision
Recall
F1
各 scene 指标
与上一轮的指标差异
是否达到 acceptance
question 定义是否变化（question 型候选；已登记 question_registry.yaml）
仍存在的问题
```

完成后将该模型当前版本：

```yaml
status: completed
```

如果 acceptance 为 false，由下一轮继续迭代。
