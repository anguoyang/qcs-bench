# qcs-bench

Public models × Qualcomm silicon, measured the same way every time. Answers two questions: which model to pick, and how fast it runs on your QCS / Snapdragon chip.

Sister repo of [rk-bench](https://github.com/anguoyang/rk-bench), same rules: one measurement protocol, one JSON source of truth, failures recorded as UNKNOWN, never filled in.

## Measurement protocol (v1.0)

- Devices: Dragonwing RB3 Gen 2 Vision Kit (QCS6490), Samsung Galaxy S25 (Snapdragon 8 Elite (SM8750)), via Qualcomm AI Hub Workbench cloud devices.
- Runtimes: tflite, precompiled_qnn_onnx; precisions: float, w8a8.
- Workbench profile job, ~100 inference runs per cell; latency shown as profile estimate (first/min run) and median over the run list
- 23 real COCO val2017 images, squash-resized to model input, RGB float32 [0,1]; same set for every quantized model
- a cell that does not compile or profile is recorded as UNKNOWN with the job id and the failure text, never filled in from another cell
- Layer split: per-layer compute_unit from the profile JSON (NPU/GPU/CPU).

## Leaderboard

### YOLOv11-N (Ultralytics, AGPL-3.0 — benchmark reference only, not shipped as a recipe)

Input 1x3x640x640 · COCO mAP —

| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | tflite | float | 134.70 ms | 19/343 | — DynamicConv2D falls back to CPU
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a8 | 5.03 ms | 341/341 | — compile jprdr9kkg / profile jgk01dnv5
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a16 | — ms | — | — UNKNOWN: compile failed, exit code 14 only (job jp1193rnp)
| Dragonwing RB3 Gen 2 Vision Kit | qnn_dlc | float | — ms | — | — UNKNOWN: compose failed on device
| Samsung Galaxy S25 | tflite | float | 2.94 ms | — | — all NPU
| Samsung Galaxy S25 | precompiled_qnn_onnx | float | 2.97 ms | — | — all NPU

### NanoDet-Plus-m 416 (Apache-2.0) — recipe: huggingface.co/anguoyang/nanodet

Input 1x3x416x416 · COCO mAP 30.4

| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | tflite | float | 90.00 ms | 60/352 | — 60 NPU + 123 GPU + 169 CPU fallback layers; profile jgd49eorg
| Dragonwing RB3 Gen 2 Vision Kit | tflite | w8a8 | 6.47 ms / 7.61 ms | 439/439 | — compile jgow04lkp / profile jgk0mxj25; quantized model mn0xxo2pz
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a8 | — ms | — | — UNKNOWN: Workbench rejected the quantized graph — float tensor /distribution_project/... remains (GFL decode arithmetic not quantized by the Hub quantizer), compile jprdqn07g
| Samsung Galaxy S25 | tflite | float | 1.47 ms | 352/352 | — all NPU
| Samsung Galaxy S25 | precompiled_qnn_onnx | w8a8 | 1.39 ms / 1.40 ms | 324/324 | — compile jp2d6vwq5

### DAMO-YOLO-S (TinyNAS L25, 640) - published at huggingface.co/anguoyang/damoyolo: the accuracy pick of this bench (+1.9 COCO mAP over the official YOLOv8-S recipe), w8a8 fully NPU-resident on both devices; ~2-3x slower than YOLOv8-S on the same silicon (latency table is the trade-off, in the open). Upstream checkpoint host is dead (OSS bucket gone); the recipe pins an HF mirror of the same file, sha256-verified

Input 1x3x640x640 · COCO mAP 46.8

| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | tflite | float | 180.35 ms / 238.80 ms | 26/186 | — 138 GPU + 22 CPU fallback layers; profile jpxznzn95
| Dragonwing RB3 Gen 2 Vision Kit | tflite | w8a8 | 15.62 ms / 17.82 ms | 214/214 | — quantize j578040lp, quantized model mq3eed2p6; profile j5mlqm1yg
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a8 | 13.27 ms / 14.78 ms | 231/231 | — profile jpyyr20rp
| Samsung Galaxy S25 | tflite | float | 7.88 ms / 7.94 ms | 186/186 | — all NPU; profile jpenk831p
| Samsung Galaxy S25 | tflite | w8a8 | 6.13 ms / 6.49 ms | 214/214 | — profile jpvnv90kg
| Samsung Galaxy S25 | precompiled_qnn_onnx | w8a8 | 2.43 ms / 2.46 ms | 234/234 | — profile j5ml3d6dg

### YOLOv8-S (Ultralytics, AGPL-3.0 — benchmark reference only, not shipped as a recipe); official Qualcomm AI Hub Models baseline

Input 1x3x640x640 · COCO mAP 44.9

| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | tflite | float | 62.35 ms / 83.19 ms | 13/259 | — 13 NPU + 224 GPU + 22 CPU fallback layers; compile jgz0r4445 / profile j578x77np
| Dragonwing RB3 Gen 2 Vision Kit | tflite | w8a8 | 5.04 ms / 6.81 ms | 286/286 | — compile jgz0r4m45 / profile j578x74np; quantized model mmx7eekly
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a8 | 6.00 ms / 6.69 ms | 285/285 | — compile jg9qwx7qp / profile jgz0r4665
| Samsung Galaxy S25 | tflite | float | 2.66 ms / 2.72 ms | 259/259 | — all NPU; compile jgdmo2erg / profile j5ml386dg
| Samsung Galaxy S25 | tflite | w8a8 | 0.92 ms / 0.95 ms | 286/286 | — compile jpenky4vp / profile jpyy6mvlp
| Samsung Galaxy S25 | precompiled_qnn_onnx | w8a8 | 1.25 ms / 1.29 ms | 288/288 | — compile jpyy6mm0p / profile jp1me6nnp

### PicoDet-M 416 (PaddleDetection, Apache-2.0) - recipe NOT published: wins COCO mAP vs NanoDet (34.4 vs 30.4) and the RB3 float cell, but its w8a8 graph falls back to CPU on RB3 and runs ~18x slower than NanoDet on the headline path; recorded for honesty. Graph exported with export.nms=false (the stock NMS-in-graph export failed all six cells: TFLite unknown-rank, QNN subgraph split)

Input 1x3x416x416 · COCO mAP 34.4

| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | tflite | float | 42.20 ms / 51.94 ms | 14/287 | — 14 NPU + 256 GPU + 17 CPU layers; profile jp42v0e8g
| Dragonwing RB3 Gen 2 Vision Kit | tflite | w8a8 | 120.69 ms / 155.84 ms | 496/696 | — 496 NPU + 192 CPU + 8 GPU - CPU fallback makes w8a8 SLOWER than its own float path; quantize j5ml3yeyg, quantized model mm5dejoyk; profile jgj8lnqx5
| Dragonwing RB3 Gen 2 Vision Kit | precompiled_qnn_onnx | w8a8 | — ms | — | — UNKNOWN: compile jp2zmk7xp failed - a BatchNormalization fused-bias tensor was left in floating point, not supported by the targeted device (same failure family as NanoDet QNN rejection)
| Samsung Galaxy S25 | tflite | float | 1.96 ms / 1.99 ms | 287/287 | — all NPU; profile j5wr9evmp
| Samsung Galaxy S25 | tflite | w8a8 | 2.42 ms / 2.44 ms | 696/696 | — all NPU; profile jp1m827lp
| Samsung Galaxy S25 | precompiled_qnn_onnx | w8a8 | 2.30 ms / 2.33 ms | 570/570 | — all NPU; profile jp3rmny3p

## GenAI — LLMs on device

Tokens per second measured on real devices via Qualcomm AI Hub Workbench profile jobs. Runtime labels the execution path (genie = QNN/Genie context binary; geniex_llamacpp = GGUF via llama.cpp) — they are different stacks and are never merged into one number.

### Qwen3-0.6B (LLM)

Checkpoint DEFAULT_W4A16 official pre-quantized

| Device | Runtime | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) |
|---|---|---|---|---|
| Dragonwing RB3 Gen 2 Vision Kit | genie | UNKNOWN | UNKNOWN | UNKNOWN | ([job](https://workbench.aihub.qualcomm.com/jobs/jp0x1z32p/)) — Hub compile rejected: --output_names is not supported for AIMET ONNX models. Pipeline bug, not a chip rejection (qualcomm/ai-hub-models#366; fix promised in v0.67.0)
| Samsung Galaxy S25 | genie | UNKNOWN | UNKNOWN | UNKNOWN | ([job](https://workbench.aihub.qualcomm.com/jobs/jp2z70o4p/)) — Identical compile rejection on the officially supported flagship - device-independent
| Dragonwing RB3 Gen 2 Vision Kit | geniex_llamacpp | UNKNOWN | UNKNOWN | UNKNOWN | — Unreachable in 0.66.0: q4_0 declared in supported_precisions but no quantization recipe ships (qualcomm/ai-hub-models#367)

## Why the same model can be 27x faster

YOLOv11-N on the RB3 (QCS6490): 134.7 ms with the default TFLite float path, 5.03 ms with w8a8 + precompiled QNN — same model, same chip. Most published numbers are the slow path. This repo exists so you pick the fast one on purpose.
