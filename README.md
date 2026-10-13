# VISTA-Bench

**Documentation:** [English](README.md) | [简体中文](README_zh.md)

A multilingual benchmark for evaluating translation from real-world images and released OCR.

<p align="center">
  <a href="https://github.com/lvbotenbest/VISTA-BENCH"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="GitHub Code"></a>
  <a href="https://huggingface.co/datasets/lvpokoto/VISTA-Bench"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset"></a>
  <a href="https://arxiv.org/abs/2609.37287"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white" alt="Paper"></a>
</p>

<p align="center">
  <img src="assets/vista_bench_framework.png" alt="VISTA-Bench framework" width="100%">
</p>

<p align="center">
  <img src="assets/rubric_case.png" alt="Image-specific rubric example" width="100%">
</p>

## Results

Overall denotes global translation quality (T), not a composite of V, T, and K.

| Model | Overall | Transport | Dining | Products | Tourism | Healthcare | Safety | Social Media | Publications | Finance | Public Services |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Image-input: closed-weight models** | | | | | | | | | | | |
| GPT-5.6 Luna | 55.88 | 51.00 | 60.03 | 53.44 | 54.38 | 61.84 | 60.94 | 58.92 | 58.59 | 58.35 | 63.55 |
| Claude Sonnet 5 | 69.43 | 64.71 | 71.78 | 65.71 | 71.14 | 74.18 | 77.94 | 67.08 | 71.38 | 78.13 | 77.82 |
| Muse Spark 1.3 | 71.56 | 64.09 | 71.98 | 70.13 | 71.99 | 77.43 | 77.59 | 73.70 | 76.86 | 72.75 | 73.95 |
| Qwen3.8 Max | 72.67 | 66.00 | 71.94 | 72.57 | 72.89 | 76.20 | 81.07 | 74.67 | 75.57 | 76.43 | 75.40 |
| Gemini 3.8 Flash | 74.20 | 69.10 | 75.62 | 74.57 | 76.22 | 79.52 | 79.12 | 72.64 | 76.34 | 78.40 | 78.17 |
| Doubao Seed 2.1 Pro | 75.12 | 68.43 | 73.48 | 75.66 | 74.24 | 77.60 | 81.89 | 77.48 | 78.38 | 80.20 | 78.87 |
| GPT-6 Astra | **77.63** | **73.78** | **76.03** | **76.11** | **77.02** | 76.98 | **83.73** | **81.63** | **80.19** | 79.70 | **79.17** |
| **Image-input: open-weight models** | | | | | | | | | | | |
| Gemma-4 E4B | 19.54 | 20.55 | 12.46 | 17.12 | 23.60 | 23.85 | 27.74 | 23.53 | 17.65 | 15.83 | 30.80 |
| GLM-4.1V 9B | 29.96 | 30.39 | 25.36 | 27.78 | 31.06 | 36.42 | 33.67 | 31.99 | 32.86 | 38.78 | 35.71 |
| GLM-5.3 Flash | 55.33 | 52.41 | 55.44 | 53.89 | 54.47 | 63.23 | 60.37 | 55.24 | 60.74 | 64.68 | 60.62 |
| DeepSeek Flash | 61.04 | 52.41 | **62.95** | 57.80 | **63.44** | **69.86** | **69.91** | 62.51 | 65.88 | **68.92** | 68.15 |
| Qwen3.8 27B | **61.11** | **56.79** | 55.63 | **60.18** | 63.22 | 68.80 | 67.59 | **63.18** | **67.32** | 65.60 | **68.52** |
| **Text-input models** | | | | | | | | | | | |
| Aya-23 8B | 47.43 | 48.97 | 41.83 | 52.95 | 51.41 | 54.14 | 49.67 | 46.79 | 46.73 | 40.91 | 48.96 |
| Gemma-3 4B IT | 55.99 | 56.22 | 52.28 | 60.13 | 57.92 | 63.03 | 58.72 | 54.22 | 55.29 | 50.11 | 58.52 |
| Qwen3 4B | 64.34 | 65.44 | 54.00 | 68.55 | 66.27 | 72.38 | 67.10 | 61.73 | 65.56 | 59.56 | 67.60 |
| HY-MT2 30B-A3B | **72.82** | **72.75** | **71.05** | **78.02** | **75.65** | **81.67** | **75.42** | **68.12** | **75.22** | **64.56** | **76.15** |

## Overview

VISTA-Bench covers 23 source languages and 10 everyday-use domains. Each image is paired with OCR, verified translations, and two evaluation rule sets: `rubric`, which retains harder criteria for stronger models, and `basic_rubric`, whose criteria are intentionally easier for smaller models. A candidate model produces a translation, a **text-only judge** evaluates it against the target-language rubric, and deterministic `scoring` produces task scores and aggregates.

> [!IMPORTANT]
> 🚀 **We will continue updating the dataset with more images, languages, and application scenarios.**

## Quick Start

Python 3.11 or later is required because the scripts use the standard-library `tomllib` module.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The code repository does not contain the dataset. Download `data/` from Hugging Face before the first run:

```bash
python -m pip install -U huggingface_hub
hf download lvpokoto/VISTA-Bench \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

### Run the benchmark

The easiest option is the interactive launcher. It asks for the API endpoint, models, evaluation track, rubric, target languages, and output directory. API keys entered interactively are hidden and are not written to the repository.

```bash
python run_benchmark.py
```

For a reproducible command-line run, store the API key in an environment variable and start with a small three-task check:

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
  --output-dir outputs/my-run
```

After the pipeline completes successfully, remove `--limit 3` to evaluate all matching tasks. Repeat `--target-lang` to select multiple targets. Add `--dry-run` to inspect the selected tasks without sending API requests.

## Unified entry-point options

| Option | Meaning |
|---|---|
| `--interactive` | Force the setup wizard; it is the default for an argument-free interactive terminal. |
| `--non-interactive` | Disable prompts for CI; all required values and the key environment variable must already exist. |
| `--api-url` | OpenAI-compatible API base URL; never embed a key in the URL. |
| `--key-env` | **Name** of the environment variable holding the API key, not the key value. |
| `--candidate-model` | Model that generates candidate translations. |
| `--judge-model` | Text-judge model; it may differ from the candidate model. |
| `--endpoint` | `chat/completions` or `responses`. |
| `--track` | `image` or `text`. |
| `--rubric-set` | `rubric`: retains selected hard rules for stronger models; `basic_rubric`: uses easier individual criteria for smaller or less capable models. |
| `--target-lang` | Filter by target language; repeat for values such as `zh-Hans` and `es`. |
| `--limit` | Run only the first N selected tasks; useful for smoke tests. |
| `--output-dir` | Directory for this run's artifacts. |
| `--resume` | Continue a compatible output directory and skip completed work. |
| `--dry-run` | Validate configuration and print the plan without making API requests. |
| `--candidate-workers` | Candidate-stage concurrency. |
| `--judge-workers` | Judge-stage concurrency. |
| `--max-attempts` | Maximum normal attempts per API stage, from 1 to 3. |
| `--timeout-seconds` | Per-request API timeout in seconds. |

Candidate and judge can normally share `--api-url`, `--endpoint`, and `--key-env`, while using different values for `--candidate-model` and `--judge-model`.

## Pipeline and tracks

1. **Task selection:** select image, source-language, and target-language directions from the current manifest.
2. **Candidate:** the image track receives the image; the text track receives released OCR only. Both produce self-segmented candidate blocks.
3. **Text-only judge:** receives source blocks, candidate blocks, target-specific criteria, and tolerances; it **never receives the image or reference translation**.
4. **Scoring:** `scoring` deterministically evaluates the judge checklist.
5. **Summary:** produces coverage, separate V/T/K direction and domain scores, and independent rankings.

The tracks are separate comparison groups:

- **Image track:** the candidate sees the image; image understanding, translation quality, and knowledge entailment can be evaluated.
- **Text track:** the candidate sees OCR only; visual criteria are removed before judging and scoring.

Image and text are separate comparison groups: their scores are **not directly comparable and must not be mixed in one leaderboard**. Both report the applicable V (visual understanding), T (translation quality), and K (knowledge entailment) dimensions separately, with no cross-dimension weighting or renormalization.

## Rubrics

### `rubric`

The primary evaluation collection, recommended for stronger models. It selects and retains hard evaluation rules to differentiate stronger model capabilities.

### `basic_rubric`

The baseline rule set for smaller or less capable models. “Basic” means that each criterion is easier to satisfy, not that the set contains fewer criteria; under the current target-language projections, `basic_rubric` has more criterion instances than `rubric`. Its release assets remain under `data/simple_rubric/` only to preserve compatibility with the existing data manifest.

The two rubric sets represent different evaluation conditions and must be aggregated and ranked separately; they must never share a leaderboard. A zero-criterion task makes no judge request and is recorded as `unscorable`; it is neither a numeric zero nor an API failure.

## Scoring

- Each expected atom is binary: `met = 1` or `unmet = 0`. Only full satisfaction is `met`; partial satisfaction must be `unmet`, and the judge does not accept `partial`.
- Criterion score = fully met expected-atom count / total expected-atom count. If any forbidden atom matches, that criterion is set to 0.
- V (visual understanding), T (translation quality), and K (knowledge entailment) are scored separately. Within each dimension, criterion importance is weighted as `core = 3`, `supporting = 2`, and `minor = 1`.
- Dimensions have no cross-dimension weights. No composite `overall` is computed, and missing dimensions are not renormalized.
- The three dimensions are aggregated and ranked independently. One `leaderboard.csv` stores V/T/K scores and three independent ranks; each rubric set still has its own leaderboard and the two sets must not be mixed.
- Null, failed, and unscorable tasks are excluded; valid numeric zeros remain included.

## Outputs and resume

`--output-dir` contains:

| File | Contents |
|---|---|
| `run.json` | Key-free run parameters, stage status, and counts; the resume compatibility record. |
| `candidates.jsonl` | Self-segmented candidate translations. |
| `candidate_failures.jsonl` | Candidate failures for audit and retry. |
| `judgments.jsonl` | Per-task checklist results from the text-only judge. |
| `scores.jsonl` | Per-task criterion details, V/T/K dimension scores, and statuses; no composite `overall`. |
| `summary.json` | Profile, rubric, coverage, V/T/K aggregation rules, and three independent ranking summaries. |
| `leaderboard.csv` | V/T/K scores and three independent ranks in one table; no composite `overall` rank. |
| `coverage.csv` | Success, failure, and unscorable coverage. |
| `scores_by_direction.csv` | Separate V/T/K source-to-target aggregates. |
| `scores_by_direction_level1.csv` | Separate V/T/K source-to-target × top-level-domain aggregates. |
| `scores_by_level1.csv` | Separate V/T/K top-level-domain macros over valid directions. |

`--resume` must point to the original output directory. The entry point reuses the same track, rubric, models, API settings, target languages, and task scope, skips valid candidate/judge records, and regenerates deterministic scores and summaries. Use a new output directory when core parameters change. Do not manually concatenate JSONL from different tracks, rubrics, or models.

## Data directories

- `data/manifest.jsonl`: sample and asset index.
- `data/images/`, `data/ocr/`, `data/verified_translations/`: images, OCR, and verified translations.
- `data/simple_rubric/`: compatibility directory backing `basic_rubric`.
- `data/rubric/`: harder criteria for stronger models.
- `script/`: candidate, judge, scoring, and summarization scripts plus schemas.

## Standalone scripts (advanced)

Most users should run `python run_benchmark.py`. Copy and edit `config.example.toml` only when custom intermediate files or pipeline research require the standalone scripts:

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

For the text track, replace the first command with `script/generate_text_candidates.py` and use `--profile text` for judge, scoring, and summarization. Never combine profiles or rubric sets in one summary.

## Citation

If you find this project useful for your research, please cite:

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
