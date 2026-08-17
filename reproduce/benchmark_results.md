# MindTS Environment Benchmark — Results

Run date: 2026-08-17 (completed 18:44 CDT)
Device: single NVIDIA RTX A4500 (20 GiB), **GPU 1 only** (`CUDA_VISIBLE_DEVICES=1`)
Script: `reproduce/scripts/univariate_detection/detect_label/Environment_script/MindTS.sh`

## Metrics comparison (origin team vs ours)

Values ×100. Ours are the means over the 42 anomaly-ratio rows of the result archive
(`reproduce/result/label/Environment/MindTS.1787010269.CS-544619.1644422.csv.tar.gz`),
the same mean aggregation the benchmark report uses.

| Dataset     | Aff-F (origin) | V-PR (origin) | V-ROC (origin) | Aff-F (ours) | V-PR (ours) | V-ROC (ours) |
| ----------- | -------------- | ------------- | -------------- | ------------ | ----------- | ------------ |
| Weather     | 82.66          | 57.48         | 82.64          | —            | —           | —            |
| Energy      | 74.37          | 50.36         | 74.44          | —            | —           | —            |
| Environment | 85.29          | 56.79         | 93.78          | **66.11**    | **51.56**   | **93.13**    |
| KR          | 90.28          | 53.15         | 89.86          | —            | —           | —            |
| EWJ         | 83.89          | 50.42         | 84.12          | —            | —           | —            |
| MDT         | 89.19          | 65.44         | 83.02          | —            | —           | —            |

### Metric definitions (as computed by this project)

- **Aff-F** — `affiliation_f` in `ts_benchmark/evaluation/metrics/classification_metrics_label.py:204`,
  affiliation-based F-score (label strategy).
- **V-PR** — `VUS_PR` (`classification_metrics_label.py:508`), volume under the PR surface;
  computed via `generate_curve` → `metricor().RangeAUC_volume` in
  `ts_benchmark/evaluation/metrics/vus_metrics.py:459/396`, with sliding window =
  2 × median anomaly-segment length.
- **V-ROC** — `VUS_ROC` (`classification_metrics_label.py:498`), volume under the ROC surface,
  same `generate_curve` machinery.

Note: V-PR and V-ROC are computed from the raw anomaly scores and are therefore constant
across the 42 threshold rows; Aff-F varies per threshold and is reported as its mean.

## Run configuration

| Hyper-parameter      | Value |
| -------------------- | ----- |
| batch_size (micro)   | 8     |
| accumulation_steps   | 8 (effective batch size **64**, matching the original script's `batch_size`) |
| num_epochs           | 1     |
| seq_len / win_size   | 72    |
| patch_size / stride  | 6 / 6 |
| mask_ratio / r       | 0.4 / 0.9 |
| d_model / d_ff / e_layers | 64 / 64 / 1 |
| enc_in_time          | 1     |
| parallel_strategy    | DP (single visible GPU → no DataParallel) |

Model: MindTS with DeepSeek-R1-Distill-Qwen-1.5B backbone (6 LLM layers used), fp32.
Gradient accumulation is implemented in `reproduce/ts_benchmark/baselines/MindTS/MindTS.py`
(`accumulation_steps` hyper-param, default 1): loss is scaled by `1/accumulation_steps` and
`optimizer.step()` runs every N micro-batches, with a remainder flush at epoch end.

## Notes / caveats

- Full run log: `reproduce/environment_run_gradacc8.log` (≈1 h 50 m total; ~43 min training, ~65 min evaluation).
- The batch-8 run **without** gradient accumulation (effective batch 8) scored Aff-F 63.38
  (its archive: `MindTS.1787003121.CS-544619.1596223.csv.tar.gz`); the accumulation run
  improved Aff-F to 66.11, moving V-ROC to 93.13 (origin: 93.78) while V-PR remains below origin (51.56 vs 56.79).
- The origin team's hardware, batch configuration and epoch count are unknown; their 3-GPU
  original config (`--gpus 0 1 2`, batch 64) does not fit this device's 20 GiB GPU
  (the full-vocabulary LM-head over 1024-token text inputs needs 55+ GiB at batch 64),
  which is the reason for micro-batch 8 + accumulation 8.
