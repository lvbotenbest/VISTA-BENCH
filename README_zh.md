# VISTA-Bench

**文档语言：** [English](README.md) | [简体中文](README_zh.md)

面向真实世界图像与公开 OCR 的多语言翻译评测基准。

<p align="center">
  <a href="https://github.com/lvbotenbest/VISTA-BENCH"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub Code"></a>
  <a href="https://huggingface.co/datasets/lvpokoto/VISTA-Bench"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset"></a>
  <a href="https://arxiv.org/abs/2609.37287"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white" alt="论文"></a>
</p>

<p align="center">
  <img src="assets/vista_bench_framework.png" alt="VISTA-Bench 框架" width="100%">
</p>

<p align="center">
  <img src="assets/rubric_case.png" alt="图像特定 rubric 示例" width="100%">
</p>

## 论文结果

<p align="center">
  <img src="assets/paper_results_table.png" alt="论文中的翻译质量结果" width="760">
</p>

<p align="center"><sub>论文表 1。Overall 表示全局翻译质量（T），并非 V、T、K 的综合分数。</sub></p>

## 概览

VISTA-Bench 覆盖 23 种源语言与 10 类日常场景。每张图像配有 OCR、人工核验翻译，以及两套评测规则：面向强模型、保留较难规则的 `rubric`，以及面向小模型、规则本身更简单的 `basic_rubric`。候选模型生成翻译后，由**只接收文本的 judge**按目标语言 rubric 检查，最后使用确定性的 `scoring` 计分并汇总。

> [!IMPORTANT]
> 🚀 **我们将持续更新数据集，进一步丰富图片数量、覆盖语种与应用场景。**

## Quick Start

需要 Python 3.11 或更高版本（脚本使用标准库 `tomllib`）。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

代码仓库不包含数据。首次运行前，从 Hugging Face 下载 `data/`：

```bash
python -m pip install -U huggingface_hub
hf download lvpokoto/VISTA-Bench \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

### 1. 无参数交互运行

统一入口会询问 track、rubric、API、模型、目标语言、任务上限和输出目录；密钥使用隐藏输入，不写入仓库。

```bash
python run_benchmark.py
```

### 2. 三任务 smoke run

先把密钥放入环境变量，再运行 3 个任务。`--target-lang` 使用数据中的语言代码；例如简体中文为 `zh-Hans`。

```bash
export VISTA_API_KEY='your-secret-value'

python run_benchmark.py \
  --non-interactive \
  --api-url https://api.example.com/v1 \
  --key-env VISTA_API_KEY \
  --candidate-model your-candidate-model \
  --judge-model your-judge-model \
  --endpoint chat/completions \
  --track image \
  --rubric-set rubric \
  --target-lang zh-Hans \
  --limit 3 \
  --output-dir outputs/smoke \
  --candidate-workers 2 \
  --judge-workers 2
```

可在发出请求前用同一命令追加 `--dry-run`，检查任务选择和运行计划。

### 3. 完整 CLI

下面展示统一入口的全部常用控制项。删除 `--limit` 即评测全部匹配任务；重复 `--target-lang` 可选择多个目标语言。

```bash
python run_benchmark.py \
  --non-interactive \
  --api-url https://api.example.com/v1 \
  --key-env VISTA_API_KEY \
  --candidate-model your-candidate-model \
  --judge-model your-judge-model \
  --endpoint responses \
  --track image \
  --rubric-set rubric \
  --target-lang zh-Hans \
  --target-lang es \
  --output-dir outputs/full-image-rubric \
  --candidate-workers 8 \
  --judge-workers 8 \
  --max-attempts 3 \
  --timeout-seconds 120
```

## 统一入口参数

| 参数 | 说明 |
|---|---|
| `--interactive` | 强制进入交互向导；无参数且终端可交互时默认启用。 |
| `--non-interactive` | 禁止提示输入，适合 CI；所需参数与密钥环境变量必须预先提供。 |
| `--api-url` | OpenAI-compatible API base URL；不要在 URL 中嵌入密钥。 |
| `--key-env` | 保存 API key 的环境变量**名称**，不是密钥值。 |
| `--candidate-model` | 生成候选翻译的模型名。 |
| `--judge-model` | 文本 judge 的模型名；可与 candidate 不同。 |
| `--endpoint` | `chat/completions` 或 `responses`。 |
| `--track` | `image` 或 `text`。 |
| `--rubric-set` | `rubric`：筛选保留难规则，推荐能力较强的模型；`basic_rubric`：规则本身更简单，推荐小模型或能力较弱模型。 |
| `--target-lang` | 按目标语言过滤；可重复，例如 `zh-Hans`、`es`。 |
| `--limit` | 只运行筛选后最前面的 N 个任务；适合 smoke test。 |
| `--output-dir` | 本次运行的产物目录。 |
| `--resume` | 从兼容的既有输出继续，跳过已完成项。 |
| `--dry-run` | 只检查配置和任务计划，不发 API 请求。 |
| `--candidate-workers` | candidate 阶段并发数。 |
| `--judge-workers` | judge 阶段并发数。 |
| `--max-attempts` | 每个 API 阶段的最大正常尝试次数，范围 1–3。 |
| `--timeout-seconds` | 单次 API 请求超时秒数。 |

Candidate 与 judge 通常可以共用 `--api-url`、`--endpoint` 和 `--key-env`，但通过 `--candidate-model` 与 `--judge-model` 使用不同模型。

## 流程与 tracks

1. **任务选择：**从当前 manifest 选择图像、源语言和目标语言方向。
2. **Candidate：**image track 输入图像；text track 仅输入公开 OCR。两者都输出 self-segmented candidate blocks。
3. **Text-only judge：**接收源文本块、candidate blocks、目标语言 criteria 与 tolerances；**永远不接收图像或参考译文**。
4. **Scoring：**`scoring` 对 judge checklist 做确定性计算。
5. **Summary：**生成 coverage、按 V/T/K 分开的方向分数、领域分数与独立排名。

两个 track 是不同的比较组：

- **Image track：**candidate 看图；可评估图像理解、翻译质量和知识蕴含。
- **Text track：**candidate 只看 OCR；视觉 criteria 在 judge 和计分前被移除。

Image 与 text 是独立比较组，分数**不可直接比较，也不能混入同一个 leaderboard**。两者都分别报告适用的 V（图像理解）、T（翻译质量）与 K（知识蕴含）维度，不做跨维度加权或重归一化。

## Rubrics

### `rubric`

主要评测集合，推荐能力较强的模型使用。该集合筛选并保留较难的评测规则，用于区分强模型的能力。

### `basic_rubric`

面向小模型或能力较弱模型的基础规则集。这里的 “basic” 表示每条规则的判定要求更简单，而不是规则数量更少；按当前目标语言投影，`basic_rubric` 的 criterion instances 反而比 `rubric` 更多。底层发布资产仍存放在 `data/simple_rubric/`，仅保留该目录名以兼容既有数据清单。

两套 rubric 代表不同评测条件，必须分别汇总和排名，不能混榜。零 criterion 任务不会调用 judge，会记为 `unscorable`；它不是数值零，也不是 API 失败。

## 评分

- 每个 expected atom 只做二值判定：`met = 1`、`unmet = 0`；只有完全满足才是 `met`，部分满足必须判为 `unmet`，judge 不接受 `partial`。
- Criterion 分数 = 完全命中的 expected atom 数 / expected atom 总数；任一 forbidden atom 命中时，对应 criterion 直接记为 0。
- V（图像理解）、T（翻译质量）与 K（知识蕴含）分别计分。每个维度内部按 criterion importance 加权：`core = 3`、`supporting = 2`、`minor = 1`。
- 维度之间不使用权重，不计算 composite `overall`，也不因维度缺失而重归一化。
- 三个维度分别聚合、分别排名；同一个 `leaderboard.csv` 同时保存 V/T/K 分数和三个独立 rank。两套 rubric 仍各自生成 leaderboard，不能混榜。
- Null、失败和 unscorable 任务排除；合法数值零仍保留。

## 输出与 resume

`--output-dir` 中包含：

| 文件 | 内容 |
|---|---|
| `run.json` | 无密钥的运行参数、阶段状态和计数；resume 的兼容性依据。 |
| `candidates.jsonl` | Self-segmented 候选翻译。 |
| `candidate_failures.jsonl` | Candidate 失败记录，便于审计和重试。 |
| `judgments.jsonl` | Text-only judge 的逐任务 checklist 结果。 |
| `scores.jsonl` | `scoring` 的逐任务 criterion 明细、V/T/K 维度分数与状态；不含 composite `overall`。 |
| `summary.json` | Profile、rubric、coverage、V/T/K 聚合规则与三个独立排名的摘要。 |
| `leaderboard.csv` | 在同一张表中保存 V/T/K 分数及三个独立 rank；不含 composite `overall` rank。 |
| `coverage.csv` | 成功、失败与 unscorable 覆盖统计。 |
| `scores_by_direction.csv` | 按源语言→目标语言分别汇总 V/T/K。 |
| `scores_by_direction_level1.csv` | 按源语言→目标语言×一级领域分别汇总 V/T/K。 |
| `scores_by_level1.csv` | 按一级领域对有效方向做等权 macro，分别汇总 V/T/K。 |

使用 `--resume` 时必须指向原输出目录。入口会复用相同的 track、rubric、模型、API 配置、目标语言和任务范围，跳过有效的 candidate/judge 记录，并重新生成确定性 score 与 summary。若核心参数不同，请使用新的输出目录；不要手工拼接不同 track、rubric 或模型的 JSONL。

## 数据目录

- `data/manifest.jsonl`：样本及资产索引。
- `data/images/`、`data/ocr/`、`data/verified_translations/`：图像、OCR 与核验翻译。
- `data/simple_rubric/`：`basic_rubric` 的底层兼容目录。
- `data/rubric/`：强模型使用的难规则集合。
- `script/`：candidate、judge、scoring、summarize 脚本及 schema。

## 独立脚本（高级用法）

一般用户应使用 `python run_benchmark.py`。只有需要自定义中间产物或研究 pipeline 细节时，才复制并编辑 `config.example.toml`：

```bash
cp config.example.toml config.toml
export VISTA_API_KEY='your-secret-value'

python script/generate_image_candidates.py \
  --config config.toml \
  --output outputs/image_candidates.jsonl \
  --failures-output outputs/image_candidate_failures.jsonl \
  --workers 8 --max-attempts 3 --resume

python script/judge.py \
  --config config.toml \
  --candidates outputs/image_candidates.jsonl \
  --rubric-set rubric \
  --profile image \
  --output outputs/image_judgments.jsonl \
  --workers 8 --max-attempts 3 --resume

python script/scoring.py \
  --judge-records outputs/image_judgments.jsonl \
  --rubric-set rubric \
  --profile image \
  --output outputs/image_scores.jsonl

python script/summarize.py \
  --scores model-a=outputs/image_scores.jsonl \
  --profile image \
  --rubric-set rubric \
  --output-dir outputs/image_summary
```

Text track 将第一条命令换成 `script/generate_text_candidates.py`，并在 judge、scoring、summarize 中使用 `--profile text`。不要跨 profile 或 rubric 合并 summary。

## 引用

如果这个项目对你的研究有帮助，请引用：

```bibtex
@misc{lv2026vistabenchbenchmarkingmultilingualimage,
  title={VISTA-Bench: Benchmarking Multilingual Image Translation with Image-Specific Rubrics},
  author={Bo Lv and Mao Zheng and Zheng Li and Fangxu Liu and Mingrui Sun and Tao Chen},
  year={2026},
  eprint={2609.37287},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37287}
}
```
