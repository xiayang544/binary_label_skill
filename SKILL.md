---
name: binary-label-skill
description: 构建 Turn 级二分类检测模型（标签标准→Regex 初筛→LLM 标注→人工复核→数据合成→BERT 微调→全量扫描→迭代）。当用户需要新建安全/合规类二分类检测器、搭建数据标注闭环，或对已有检测模型进行 FP/FN 归因与迭代时使用。
---

# Binary Label Skill

## 1. 目标

本 Skill 用于构建和持续迭代 Turn 级二分类检测模型。

给定标签定义、部分人工标注数据、业务场景、上下文轮数配置和候选检测模型（BERT 类或 laya 零训练类，见 §17），完成：

1. 标签标准及细分子场景设计；
2. Regex 初筛规则设计与迭代；
3. 大模型辅助标注；
4. 人工复核与标签标准校准；
5. 数据合成；
6. BERT 微调、超参数搜索和阈值搜索；
7. 全量数据扫描；
8. FP/FN 归因；
9. 数据回流和模型迭代；
10. 版本化保存所有数据、标准、Prompt、Regex、模型和评测结果。

本 Skill 分为两个工作流：

* 首次构建：读取 `workflows/cold_start.md`
* 后续迭代：读取 `workflows/iteration.md`

---

## 2. 核心检测定义

检测粒度始终为 Turn 级。

目标是判断当前 Turn 中模型输出的内容属于：

* `positive`
* `negative`

上下文仅用于辅助理解当前 Turn，不改变最终检测粒度。

每条正例必须同时具有一个 `scene_id`，表示所属正例细分子场景。

---

## 3. 用户输入

所有任务配置统一维护在根目录 `task.yaml`。

完整字段以 `templates/task.yaml` 为准，首次冷启动时必须按该模板初始化。关键字段结构：

```yaml
task:
  name: ""
  label_definition: ""
  business_scene: ""
  # 可选：wecom / outbound_call / rider_assistant / other

input:
  seed_annotation_path: ""
  online_data_path: ""

context:
  llm_label_turns: null
  bert_turns: 1
  regex_turns: 10

models:
  candidates:
    h256:
      enabled: true
    h288:
      enabled: true
    macbert:
      enabled: true
    # 零训练 laya 候选（完整约定见 §17）：
    # type: laya 时跳过训练类步骤，只做 threshold 搜索
    laya_multilingual:
      enabled: false
      type: laya

acceptance:
  precision_min: 0.90
  recall_min: 0.80
```

`task.yaml` 只保存任务级、跨版本稳定的配置；当前版本、执行状态、当前最佳模型等动态信息只写入各版本的 `status.yaml` 和 `manifest.yaml`，禁止写入 `task.yaml`。

上下文轮数全部由人工给定。

Skill 禁止自行判断“简单任务”或“复杂任务”，也禁止自行修改上下文轮数。

---

## 4. 脚本约定

通用脚本已随 Skill 提供，位于 `scripts/`（入口）与 `src/`（共享库）：

| 脚本 | 用途 | 对应 workflow 步骤 |
| --- | --- | --- |
| `scripts/run_regex_prefilter.py` | Regex 初筛线上数据 + 回归测试 | cold_start §3 / iteration §13 |
| `scripts/label_llm.py` | LLM 打标（冷启动单模型 / 迭代多模型分别调用，`--save-sample` 落盘采样快照） | cold_start §5 / iteration §3 |
| `scripts/llm_vote.py` | 多 LLM 打标结果投票合并 | iteration §3 |
| `scripts/export_review_input.py` | 生成人工复核输入表（含 BERT 预测 / vote 列 + 人工填写列） | cold_start §6 / iteration §4 |
| `scripts/import_human_review.py` | 人工复核 Excel 导入与校验 | cold_start §6 / iteration §4 |
| `scripts/build_gold.py` | 人工核验样本全量合入全局 Gold（按 `sample_id` 去重；`--mode sample` 分层抽样控制规模；`--update-labels --reason` 需求变更改标并留审计） | cold_start §6 / iteration §4、6 |
| `scripts/generate_synthetic.py` | 数据合成（场景补充 / hard case 分层） | cold_start §9 / iteration §8 |
| `scripts/build_dataset.py` | 构建 train / validation / regression_eval（含 vote 一致样本采纳） | cold_start §10 / iteration §9 |
| `scripts/train_bert.py` | BERT 微调（单组超参，供搜索循环调用） | cold_start §11 / iteration §10 |
| `scripts/hparam_search.py` | 超参网格搜索编排（支持多卡分片） | cold_start §11 / iteration §10 |
| `scripts/tune_threshold.py` | validation 上 threshold 搜索（BERT / laya 均支持） | cold_start §11 / iteration §10 |
| `scripts/select_model.py` | 单模型统一评估（validation/regression/gold） | cold_start §12 / iteration §11 |
| `scripts/evaluate.py` | 总体 + 子场景指标、FP/FN 明细、acceptance（含 min_improvement） | cold_start §12 / iteration §11、14 |
| `scripts/classify_errors.py` | FP/FN LLM 辅助归因（error_type 初稿） | iteration §5 |
| `scripts/full_scan.py` | Regex 初筛 + BERT 精筛全量扫描（支持分片、`--since/--until/--append` 增量） | cold_start §13 / iteration §13 |
| `scripts/sample_iteration_data.py` | 迭代候选池采样 | iteration §2 |

脚本约定：

1. 均在项目根目录（含 `task.yaml` 的目录）执行，通过 `--root`（默认 `.`）、`--version` 与 `--model` 定位。**`--model` 一律指检测器**（task.yaml models.candidates 的 key，即流水线归属）；执行打标/合成/归因的大模型一律用 `--llm` 参数。`--model` 缺省时取唯一 enabled 候选，多个 enabled 时必须显式指定。所有产物写入 `<model>/<version>/`（gold 除外，全局）；
2. 输出一律 JSONL + Excel 双格式（缺 openpyxl 时自动降级 CSV），并自动登记到 `status.yaml` 的 `artifacts`；
3. 上下文轮数一律读 `task.yaml`，脚本不提供修改轮数的参数；
4. 训练与推理共用 `src/text_utils.build_bert_text`，保证拼接逻辑一致；
5. LLM 接入读 `task.yaml` 的 `llm_api`（base_url / api_key / gateway_names），缺省用环境变量 `LLM_BASE_URL` / `LLM_API_KEY`；打标、合成（`generate_synthetic.py`）、归因（`classify_errors.py`）共用该配置；
6. 若某轮 Regex 无法用 `regex_config.yaml` 描述，仍允许在 `standard/regex/regex_prefilter.py` 存放定制逻辑；
7. 修改脚本后必须重新执行受影响步骤的产物校验；
8. 候选配置含 `type: laya`（缺省视为 BERT 类）时为**零训练候选**，走 §17 约定：训练/超参搜索/合成类步骤跳过，打分结果以 `laya_prob` 字段落盘，threshold 搜索沿用 `tune_threshold` 的 objective 与 `training.threshold_search` 配置。

运行环境：需要 `pandas / pyyaml / openpyxl / torch / transformers / sklearn / openai`，可参考内部已验证的 Python 3.12 环境。laya 候选额外需要 `laya` 包（本地源码见 §5，`pip install laya` 或将本地 checkout 加入 `PYTHONPATH`）。

---

## 5. 固定资源地址

线上全量数据默认地址：

```text
/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/<data_owner>/data/online_data
```

候选模型默认地址：

```text
macbert:
/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/<user>/huggingface.co/hfl/chinese-macbert-base

h256:
/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/<user>/huggingface.co/hfl/minirbt-h256

h288:
/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/<user>/huggingface.co/hfl/minirbt-h288

laya_multilingual（type: laya 候选）:
源码：
/mnt/dolphinfs/hdd_pool/docker/user/<internal_mount>/<user>/safety/laya
checkpoint：task.yaml 未显式指定 path 时，从 HF Hub 下载
（convaiinnovations/laya 仓库 multilingual 子目录，等价独立仓库 convaiinnovations/laya-multilingual）
```

路径中的 `<internal_mount>` / `<user>` / `<data_owner>` 为占位符，部署时按实际内部挂载环境替换。

如果 `task.yaml` 显式指定其他地址，以 `task.yaml` 为准。

---

## 6. 数据格式约束

数据字段统一按照 `data_schema.yaml` 执行。

所有进入正式数据集的数据至少包含：

```text
sample_id
conversation_id
turn_id
current_turn
context
label
scene_id
data_source
annotation_source
version
data_date
```

字段说明：

* `label`：`positive` 或 `negative`
* `scene_id`：正例必填，负例允许为空
* `data_date`：样本所属线上数据日期（YYYY-MM-DD）。线上全量数据按日期文件增长（如 `2026-05-01_recruitment_full_cc2.json`），初筛与扫描产物必须携带该字段；Regex 初筛与全量扫描支持 `--since/--until` 按日期增量处理，初筛支持 `--append` 将新日期的召回合并进已有 `recalled.jsonl`（按 sample_id 去重）
* `data_source`：

  * `real`
  * `synthetic`
* `annotation_source`：

  * `human`
  * `llm`
* `context`：按照 `task.yaml` 中指定的上下文轮数生成
* `version`：产生该数据的版本，如 `v_0`、`v_1`

合成数据额外记录：

```text
generator_model
synthesis_prompt_version
```

大模型标注数据额外记录：

```text
label_model
label_prompt_version
```

人工复核完成后记录：

```text
human_verified: true
```

进入 Gold 的样本额外记录：

```text
dataset_split: gold
```

需求变更改标过的 Gold 样本额外记录（见 §8）：

```text
label_history（原 label、原版本、变更版本、变更原因、时间等审计链）
```

---

## 7. Excel 与 JSONL 规则

数据原则上同时提供 Excel 和 JSONL 两种表示。

JSONL 用于程序处理和版本复现。

Excel 用于人工查看、人工复核和结果交付。

人工复核流程为：

```text
JSONL
→ 导出 review_input.xlsx
→ 人工填写 review_done.xlsx
→ import_human_review.py
→ 生成规范化 review_result.jsonl
→ 同步生成 review_result.xlsx
```

后续程序不得直接依赖人工编辑后的原始 Excel，而应读取标准化后的 JSONL。

---

## 8. Gold 数据规则

Gold 数据统一存放（全局资产，不属于任何单独版本）：

```text
data/gold/
├── gold_positive.xlsx
├── gold_positive.jsonl
├── gold_negative.xlsx
└── gold_negative.jsonl
```

收纳策略：

* 只收纳**人工核验的真实数据**（`data_source=real` 且 `human_verified=true`）；
* 每一轮人工复核完成后（冷启动 §6、迭代 §4），本轮新增的人工核验样本**全部合入** Gold：`build_gold.py` 默认 `--mode all` 全量采纳，按 `sample_id` 去重（label 相同的已有样本自动跳过）；需要控制 Gold 规模时可退回 `--mode sample`（正例按 scene_id 分层抽样、易混淆负例优先）；
* **需求变更**（标签标准演进，如某类话术从 positive 改判 negative）允许修正历史 Gold label，但必须通过脚本执行并留审计：

```text
scripts/build_gold.py --version <v_k> --model <model> \
  --input <需求变更人工重标后的 review_result.jsonl> \
  --update-labels --reason "<变更说明>"
```

  约束：

  * 禁止直接手改 `data/gold/` 的 jsonl / xlsx；
  * `--update-labels` 必须携带 `--reason`；改标样本在 `label_history` 中保留完整审计链（原 label、原版本、审核人、变更版本、原因、时间）；
  * label 变化的样本自动在 `gold_positive` / `gold_negative` 之间迁移；
  * 未加 `--update-labels` 时，输入样本 label 与 Gold 现值冲突会被跳过并逐条告警；
  * 改标即评估基准变化：当轮评估与验收（cold_start §12 / iteration §11、14）必须基于改标后的 Gold 重新执行，与上一版本 acceptance 的对比结论可能失效。

仍然禁止：

* 将 Gold 数据加入训练集；
* 根据单条 Gold 错误样本生成训练数据；
* 将 Gold 数据复制到某一轮训练集中。

Gold 用于独立质量评估（不参与训练、threshold 与超参调优）。

---

## 9. 版本与目录规则

**每个候选模型是一条完全独立的流水线**：`task.yaml` 可启用 1 个或多个模型，各模型独立执行「冷启动 → 迭代」全流程，版本链、标签标准、标注数据、训练产物互不干扰。单次工作流执行只针对一个模型（多个 enabled 时必须显式指定 `--model`）。

```text
root/
├── task.yaml                  任务级配置（启用 1 个或多个候选模型）
├── data/gold/                 全局 Gold（唯一跨模型共享资产：人工核验真实数据全量累积，需求变更可改标留审计）
├── h256/                      模型 h256 的独立流水线
│   ├── v_0/
│   │   ├── standard/          标签标准（scenes/regex/prompt）
│   │   ├── prefilter/         Regex 初筛结果
│   │   ├── labeling/          LLM 打标 + 人工复核 + 归因分析
│   │   ├── synthesis/         合成数据
│   │   ├── dataset/           train / validation / regression_eval
│   │   ├── training/          超参搜索产物（cfg<i>/、tuning_trials.csv、best_*）；laya 候选仅存 threshold 搜索产物（见 §17）
│   │   ├── metrics/           指标、selected_model、acceptance、FP/FN 明细
│   │   ├── scan/              全量扫描结果
│   │   ├── manifest.yaml
│   │   ├── status.yaml
│   │   └── report.md
│   ├── v_1/ ...
│   └── question_registry.yaml question 定义登记表（laya/kev 等 question 型候选，见 §18；BERT 类不需要）
├── h288/v_0/ ...              模型 h288 的独立流水线（版本号独立计数）
└── macbert/v_0/ ...
```

规则：

* 模型目录名默认取 `task.yaml` `models.candidates` 的 key（如 `h256`）；如需 `h_256` 风格，在候选配置中加 `dir: "h_256"`。
* **各模型版本号独立计数**：`h256/v_3` 与 `macbert/v_1` 可以并存，互不对应。
* `data/gold/` 是唯一跨模型共享的资产：任何模型的流水线人工复核产生的核验样本都合入到这里（追加去重、需求变更改标均全局生效），供所有模型的评估使用；除此之外模型之间不共享任何状态。
* 不存在跨模型选择：各模型各自迭代到自己的 acceptance，最终通过各自的 `report.md` 人工对比。

首次冷启动固定使用 `v_0/`（该模型的），第一次迭代开始使用 `v_1/`，之后依次 `v_2/ ... v_k/`。

**冻结边界**——历史版本中禁止覆盖或修改的是复现性资产：

```text
standard/    标签标准、Regex、Prompt（版本演进的对象）
dataset/     train / validation / regression_eval
training/    模型与超参搜索记录
metrics/     指标与 acceptance
manifest.yaml / report.md
```

而以下两个目录是对线上数据的**视图**，允许在同一版本上随数据增长刷新（见下节）：

```text
prefilter/   Regex 初筛结果
scan/        全量扫描结果
```

如果该模型当前最大版本为 `v_k` 且状态为 `completed`，下一次迭代必须创建 `v_{k+1}`。

新版本允许继承上一版本的标签标准、Prompt、Regex 配置和训练数据，但必须在新版本目录中保存完整快照。

### 数据增长时的版本内刷新

线上全量数据随日期增长。**数据变化不构成新版本**——版本号追踪的是「标准/Regex/模型/数据集」组合，数据增长是输入侧事件。在当前版本 `v_k` 上刷新视图即可：

```text
# 1. 增量初筛：只处理新日期，合并进已有 recalled.jsonl（按 sample_id 去重）
scripts/run_regex_prefilter.py --version v_k --model <model> --since <上次处理到的次日> --append

# 2. 用 v_k 已有模型对新召回样本重扫（增量合并）：
scripts/full_scan.py --version v_k --model <model> \
  --model-dir <v_k selected best_model> --threshold <v_k threshold> \
  --since <上次处理到的次日> --append

# 3. （可选）人工抽查看新日期的 union_positive 是否出现新模式/新场景
```

规则：

* 刷新后 `v_k/scan/` 即为该版本模型对**最新线上数据**的完整视图，下一次迭代 `v_{k+1}` 的 `sample_iteration_data.py` 读它即可；
* `--append` 只用于「数据增长 + Regex/模型/threshold 不变」。若 Regex 规则或模型/threshold 发生了变化，必须**去掉 `--append` 全量覆盖重扫**（旧结果的命中字段和分数已过时）；
* 增量扫描前应先做增量初筛（Regex 未召回的样本不进入 BERT，跳过初筛会漏掉新数据中本可召回的部分）；
* 若刷新后发现新日期数据里出现大量新模式 FP/FN，说明标准需要演进——走正常迭代流程创建 `v_{k+1}`，在 `v_{k+1}` 中更新标准与 Regex。

---

## 10. Standard 版本规则

以下内容共同组成一轮完整标签标准：

```text
standard/
├── label_definition.md
├── scenes.yaml
├── boundary_cases.jsonl
├── regex/
├── llm_label_prompt.md
└── synthesis_prompts/
```

Standard 随模型流水线版本化，存放在：

```text
<model>/v_k/standard/
```

这些内容必须随版本冻结。

例如 `h256/v_1/standard/` 表示 h256 流水线 v1 使用的完整标签标准。

如果 v2 修改了子场景、边界、Prompt 或 Regex，只允许修改：

```text
<model>/v_2/standard/
```

禁止回写 `<model>/v_1/standard/`。

---

## 11. Regex 规则

Regex 属于标签标准的一部分，允许每轮迭代。

每轮保存：

```text
standard/regex/
├── regex_config.yaml
├── regex_cases.jsonl
└── regex_report.yaml
```

Regex 通用执行逻辑放在：

```text
src/regex_engine.py
```

入口脚本为：

```text
scripts/run_regex_prefilter.py
```

若某轮 Regex 无法通过配置描述，允许额外保存：

```text
standard/regex/regex_prefilter.py
```

Regex 每轮更新后必须重新执行历史 `regex_cases.jsonl` 回归测试。

---

## 12. 数据集角色

正式数据集分为：

### train

用于 BERT 参数训练。

允许包含：

* 大模型标注数据；
* 合成数据；
* 其他明确允许进入训练集的数据。

### validation

用于：

* 超参数选择；
* 模型选择；
* threshold 调优。

禁止因为某一次模型效果不好而直接修改 label。

### regression_eval

保存迭代过程中人工确认的真实 FP/FN。

真实 FP/FN 默认只进入 `regression_eval`，不进入训练集。

如果需要训练类似 hard case，只通过大模型生成对应合成数据，再加入 train。

### gold

位于根目录 `data/gold/`。

收纳全部人工核验真实数据，随每轮人工复核累积增长；需求变更时按 §8 通过 `build_gold.py --update-labels` 改标（留审计）。

不参与训练与任何调优，仅用于独立质量评估与验收。

---

## 13. Manifest 与 Status

每个 `<model>/v_k/` 必须包含：

```text
manifest.yaml
status.yaml
```

status 只描述该模型该版本的进度：

* `current_step`：精确到 workflow 小节的步骤 ID，如 `cold_start:step_06_human_review`；
* `status`：当前状态；
* `completed_steps`：已完成步骤 ID 列表；
* `artifacts`：各步骤关键产物的存在性和规模检查表；
* `pause_info`：人工暂停时需要人工处理的文件路径；
* `model`：所属模型 key。

`manifest.yaml` 描述该版本使用了什么，字段结构以 `templates/manifest.yaml` 为准，包括：

* parent version；
* 标签标准；
* Regex；
* Prompt；
* 数据集；
* 模型；
* 数据来源；
* 关键配置。

Skill 启动时必须：

1. 确定目标模型（显式指定，或唯一 enabled 候选）；
2. 查找该模型当前最大版本 `v_k`；
3. 如果 `v_k` 未完成，从其 `status.yaml` 的 `current_step` 恢复，禁止重新执行 `completed_steps` 中产物完整的步骤；
4. 如果 `v_k` 已 `completed`，按版本规则创建 `v_{k+1}`。

禁止重复已经成功完成且产物完整的步骤。

---

## 14. 人工暂停规则

以下阶段必须暂停，不允许自动继续：

```text
waiting_standard_review
waiting_human_review
```

进入人工暂停状态时：

1. 生成待复核 Excel；
2. 更新 `status.yaml`；
3. 明确记录需要人工处理的文件路径；
4. 停止当前工作流。

只有检测到对应人工完成文件后，才允许继续执行。

禁止自行模拟人工确认。

---

## 15. 工作流路由

启动时先确定目标模型（显式指定 `--model`；task.yaml 只启用一个候选时可缺省）。

在该模型的版本链上查找当前最大版本 `v_k`：

1. 如果该模型不存在任何版本目录，读取并执行 `workflows/cold_start.md`，创建 `<model>/v_0`；
2. 如果 `v_k` 存在且 `status.yaml` 尚未达到 `completed`，读取其 `current_step`，在对应 workflow 中从该步骤恢复（`v_0` 使用 `workflows/cold_start.md`，`v_1` 及之后使用 `workflows/iteration.md`）；
3. 如果 `v_k` 已 `completed`，读取并执行 `workflows/iteration.md`，创建 `v_{k+1}` 并从步骤 1 开始。

task.yaml 启用多个模型时，各模型独立执行上述路由，互不干扰；不合并版本、不跨模型比较。

禁止在该模型 `v_k` 未完成时创建 `v_{k+1}`。

---

## 16. 最终交付要求

每轮必须至少保留：

* 当前标签标准；
* 当前 Regex；
* 当前 Prompt；
* LLM 标注数据；
* 人工复核数据；
* FP/FN 分析；
* 训练集；
* validation；
* regression_eval；
* 合成数据；
* 模型；
* threshold；
* 全量扫描结果；
* 模型指标；
* 场景指标；
* `report.md`；
* 日志。

任何一项关键产物缺失，不应将该版本标记为 `completed`。

---

## 17. Laya 候选（零训练检测器）

候选配置含 `type: laya` 时，该候选为**零训练检测器**：不做合成、不微调、不跑超参网格，检测能力来自 laya checkpoint 的非自回归决策输出（`noul` 型问题给出正例校准概率）。版本目录布局、版本号规则、人工暂停点、验收标准与 BERT 候选完全一致。

### 17.1 task.yaml 声明

```yaml
models:
  candidates:
    laya_multilingual:
      enabled: true
      type: laya              # 必填，缺省视为 BERT 类
      variant: multilingual   # english / multilingual / typed-decisions
      path: ""                # 本地 checkpoint 目录；空则从 HF Hub 下载（地址见 §5）
      max_len: 8192           # 长文档必须显式传：multilingual 默认 1024 截断
      # dir: "laya_ml"        # 可选，模型目录名覆盖（缺省取 key）
```

上下文轮数复用 `context.bert_turns`（保证与 BERT 候选、LLM 标注看到同一上下文视图）。

### 17.2 与 BERT 流水线的步骤差异

| 工作流步骤 | BERT 候选 | laya 候选 |
| --- | --- | --- |
| cold_start step_08/09（合成） | 执行 | **跳过**（无训练集需求） |
| cold_start step_10（数据集） | train/validation/regression_eval | 只构建 validation + regression_eval |
| cold_start step_11（训练+超参+阈值） | 完整执行 | 只做 threshold 搜索（对 validation 上的 `laya_prob`） |
| iteration step_08/09/10（合成/回流/重训） | 执行 | **跳过**；FP/FN 修复走 17.4 |
| 其余步骤（标准/Regex/LLM 打标/复核/归因/评估/扫描/验收） | 执行 | 照常执行 |

跳过的步骤不写入 `completed_steps`；`status.yaml` 按「该类型候选实际应执行的步骤」判定完成度。

### 17.3 打分与 threshold 约定

1. **state 构造**：按 `context.bert_turns` 用 `src/text_utils` 的统一拼接逻辑生成纯文本（context + current_turn），训练与推理不共用 `build_bert_text` 之外的任何拼接实现；
2. **question 配置**：每版本冻结一份 `standard/laya_question.yaml`，固定一个 `noul` 型问题，`instructions` 从 `label_definition.md` 派生（可附 `criteria` 说明边界）；该文件属于 standard 的一部分，随版本演进、禁止回写历史版本；
3. **打分落盘**：推理调用 `Router(models={variant: path}).predict(state, question)`，取 `result["answers"][<q>]["noul"]` 作为正例概率，写入 `laya_prob` 字段（validation 打分结果与 scan 结果均携带，并登记到 `status.yaml` 的 `artifacts`）；通用脚本（`tune_threshold.py` / `evaluate.py` / `select_model.py` / `full_scan.py` / `hparam_search.py`）已通过 `src/laya_scorer.py` 直接支持 laya 打分，无需手写定制脚本；
4. **threshold 搜索**：网格读 `task.yaml` `training.threshold_search`，objective 与 `tune_threshold.py` 一致（默认 `max_recall_given_precision`，受 `acceptance.precision_min` 约束），产物写入 `<model>/v_k/training/`：`threshold_search.csv` + `best_threshold.yaml`（无 `cfg<i>/`、无 `best_model/`）；
5. **selected_model 语义**：`metrics/selected_model.yaml` 记录 checkpoint 引用（path 或 Hub 仓库+revision）、`laya_question.yaml` 的相对路径、best_threshold；全量扫描（`full_scan.py` 语义）按此三元组执行，`--model-dir` 对 laya 候选指 checkpoint 目录。

### 17.4 迭代修复手段

laya 候选无训练集，FP/FN 的修复手段限定为（均在 `v_{k+1}` 中执行）：

1. 修改 `standard/laya_question.yaml` 的 `instructions` / `criteria` 措辞（对应归因类型 `boundary_definition`、laya 场景下 `bert_capacity` 的等价物是「question 表达力不足」），并按 §18 在 `question_registry.yaml` 登记变更；
2. 调整 threshold（配合重搜）；
3. 修改 Regex（同 BERT 候选规则，含回归测试）。

真实 FP/FN 照常回流 `regression_eval`，但**不触发 hard case 合成**（合成数据无处消费）。

### 17.5 交付物差异

§16 清单中「模型」「训练集」「合成数据」三项对 laya 候选分别替换为：checkpoint 引用 + `laya_question.yaml` + best_threshold、（无）、（无）；其余交付物不变；另按 §18 维护 `<model>/question_registry.yaml`（question 定义登记表，交付检查项之一）。

---

## 18. Question 登记表（question_registry.yaml）

基于 laya 的检测器（零训练 §17 或微调类）以及 kev 等其他以 question 形式
定义标签的候选，其 question 定义（instructions + 选项 criteria + label 表面形式）
是标签标准在模型侧的直接载体。为避免定义以常量形式分散在各版本脚本中
难以追溯，约定**每条候选流水线在模型目录根部维护一份登记表**：

```text
<model>/question_registry.yaml     # 如 laya/question_registry.yaml、kev/question_registry.yaml
```

### 18.1 登记内容

每个版本一条记录，至少包含：

```yaml
current: v_k            # 当前最新版本
versions:
  v_k:
    source: <该版本 question 定义所在脚本/配置文件路径>   # 有多个副本时逐一列出
    instructions: "<问题指令文本>"
    criteria:
      positive: "<正例选项描述>"
      negative: "<负例选项描述>"
    label_surface_forms: [是/否, A/B]   # 实际使用的表面形式；正例恒为 option index 0 时注明
    change_vs_prev: "<相对上一版的变更说明；无变化写「无（沿用 vX 原文）」>"
    input_shape: <输入形态>             # utt（仅判定轮）/ ctx（含上下文）等
    # 可选：data_baseline、selected、note
```

### 18.2 登记规则

1. **只追加不回写**：新版本修改 instructions/criteria/label 形式时，在 `versions`
   下追加记录；禁止修改历史版本记录（与 §10 standard 冻结规则同构）；
2. **版本未动也要登记**：question 沿用上版原文时，登记一条 `change_vs_prev: 无`
   的记录并注明来源版本，保证每个版本在表中都有对应条目、可查；
3. **source 必须可定位**：登记定义所在的脚本/配置文件路径；同一文本复制在多个
   脚本（prep/eval/scan）时逐一列出，并建议改为单一来源文件消除复制；
4. **打分口径一致**：训练与推理/扫描必须使用同一份定义；评估、扫描脚本中的
   question 文本与本表登记值不一致时，以本表为准修正脚本；
5. **在 status.yaml 登记 event**：新建或更新登记表时在 `event_log` 记一笔。

### 18.3 与 standard 的关系

- 零训练候选（§17）：question 已冻结在 `standard/laya_question.yaml`，
  登记表额外汇总跨版本演进（source 指向各版本 standard 文件）；
- 微调类候选：question 可能是脚本常量（历史项目）或 standard 配置文件，
  登记表是唯一的跨版本汇总视图；后续版本应逐步把定义收敛到
  `<model>/v_k/standard/` 下的单一来源文件，登记表 source 指向它；
- 标签语义的权威来源仍是 `standard/label_definition.md` + `scenes.yaml`，
  question 文本是它们在模型侧的浓缩表达；两者冲突时先修 standard 再同步 question。

### 18.4 现有项目落位

内部安全类项目（如隐私合规、法律法规违规检测）的 `laya/`、`kev/` 目录已按本约定
建立登记表（2026-09-30）。
