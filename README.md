# SoccerMaster: A Vision Foundation Model for Soccer Understanding
Official implementation of **SoccerMaster: A Vision Foundation Model for Soccer Understanding** (CVPR 2026 Oral)

[*Haolin Yang*](https://haolinyang-hlyang.github.io/),
[*Jiayuan Rao*](https://jyrao.github.io/),
[*Haoning Wu*](https://haoningwu3639.github.io/),
[*Weidi Xie*](https://weidixie.github.io/)


<div style="line-height: 1;">
  <a href="https://haolinyang-hlyang.github.io/SoccerMaster/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website🌐-SoccerMaster-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/pdf/2512.11016" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv📄-SoccerMaster-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

> **This is a fork.** Everything above, plus `codes/` and the Quick Start / Data Pipeline
> sections below, is the original SoccerMaster release by Haolin Yang, Jiayuan Rao, Haoning Wu
> and Weidi Xie — see the [upstream repository](https://github.com/haolinyang-hlyang/SoccerMaster)
> and [paper](https://arxiv.org/pdf/2512.11016) for figures, checkpoints and citation.
>
> This fork adds a tactical layer on top of SoccerMaster's game-state output, in `pipeline/` and
> `multi-agent/`: visible-topology analysis, two-stage evidence-grounded Chinese commentary,
> CosyVoice narration, rendered tactical effects, and a two-arm multi-agent tactic-recognition
> experiment. Those parts are not from the paper and are not peer reviewed.

## Repository layout

| 目录 | 内容 |
|---|---|
| `codes/` | 上游 SoccerMaster：tracklab、sn-gamestate、SAM2、预训练代码 |
| `pipeline/` | 本 fork：raw video → game state → 文字解说 → 语音 → 特效 |
| `multi-agent/` | 本 fork：战术识别的两条臂 —— A+B+rule→C 与 C-only |
| `scripts/` | 各阶段的命令行入口 |
| `benchmark/` | 两段式解说的人工输入（见 `benchmark/README.md`） |
| `tests/` | 离线测试，不打网络 |

## Quick Start

### 1. Environment Setup

Create a conda environment and install dependencies:

```bash
conda create -n tracklab_release python=3.10.16
conda activate tracklab_release

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install setuptools==78.1.1
cd ./codes/sn-gamestate
pip install -e .
cd ../tracklab
pip install -e .
cd ../sam2
pip install -e .

pip install albumentations==1.4.19
pip install git+https://github.com/huggingface/transformers
pip install accelerate==1.8.1
pip install "qwen-vl-utils[decord]==0.0.8"
pip install flash-attn --no-build-isolation
```

### 2. Download Pretrained Models

First, download the SigLIP2 backbone model to the pretrained models directory:

```bash
pip install huggingface_hub
hf download google/siglip2-large-patch16-512 --local-dir /data/zhangzheng/pretrained_models/google/siglip2-large-patch16-512
```

### 3. Download SoccerMaster Checkpoints

Clone the SoccerMaster model checkpoints from Hugging Face:

```bash
cd /data/zhangzheng/pretrained_models
mkdir SoccerMaster
hf download xleprime/SoccerMaster --local-dir SoccerMaster
```

After completing these steps, the SigLIP2 and SoccerMaster checkpoints live under `/data/zhangzheng/pretrained_models/`:

```
/data/zhangzheng/pretrained_models/
├── google/
│   └── siglip2-large-patch16-512/
└── SoccerMaster/
```

### 4. Prepare Pretrained Models

Please organize `/data/zhangzheng/pretrained_models/` as follows:

```
/data/zhangzheng/pretrained_models/
├── calibration
│   ├── mean.npy
│   ├── pnl_SV_kp
│   ├── pnl_SV_lines
│   ├── Radar.png
│   ├── std.npy
│   ├── SV_kp
│   └── SV_lines
├── google
│   └── siglip2-large-patch16-512
├── jn
│   ├── Qwen2.5-VL-72B-Instruct
│   └── Qwen2.5-VL-7B-Instruct
├── legibility
│   └── legibility_resnet34_soccer_20240215.pth
├── reid
│   ├── hrnetv2_w32_imagenet_pretrained.pth
│   └── prtreid-soccernet-baseline.pth.tar
├── SoccerMaster
│   └── ...
└── yolo
    └── yolo_v8x6_finetuned.pt
```

**Download instructions:**

- **calibration & reid**: Follow the instructions from [SoccerNet Game State Recognition](https://github.com/soccernet/sn-gamestate) to obtain the calibration and ReID pretrained models.
- **google**: Download `siglip2-large-patch16-512` from [Hugging Face](https://huggingface.co/google/siglip2-large-patch16-512).
- **legibility**: Download `legibility_resnet34_soccer_20240215.pth` from [Google Drive](https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing), following the [jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) project.
- **jn**: Download Qwen2.5-VL models from the [Qwen2.5-VL Collection](https://huggingface.co/collections/Qwen/qwen25-vl) on Hugging Face.
- **yolo**: Move `yolo_v8x6_finetuned.pt` to `/data/zhangzheng/pretrained_models/yolo/`. This file can be obtained from [https://huggingface.co/xleprime/SoccerMaster](https://huggingface.co/xleprime/SoccerMaster).

The dataset lives under `data/GSR-2025/`. Splits come from `sequences_info.json`; directory parents do not encode the split. Point elsewhere with `PipelineConfig(gsr_dataset_root=...)`.

### 5. API Keys (tactical layer only)

The game-state pipeline above needs no keys. Commentary and multi-agent recognition call an
external video model, so copy `.env.example` to `.env` and fill in what you use:

| 变量 | 用途 |
|---|---|
| `ARK_API_KEY` | Doubao（火山方舟）。`--provider doubao` 必需 |
| `GEMINI_API_KEY` | Gemini。`--provider gemini` 必需；`GEMINI_API_KEY1/2/3` 可选，用于多 key 轮换 |
| `ARK_BASE_URL` / `ARK_RESPONSES_MODEL` | 可选，覆盖 Ark 的默认地址与模型 |
| `DOUBAO_TTS_*` | Stage 3 语音合成 |

每个会把视频上传到外部 API 的命令都要显式加 `--allow-external-upload`，避免误传。

## Data Pipeline

### Input Data Format

The data pipeline follows the [SoccerNet Game State Recognition (GSR)](https://github.com/soccernet/sn-gamestate) format. You need to organize your input video frames into the following directory structure. The audited layout uses `SNGS-001` through `SNGS-200`; one sequence is shown below:

```
data/GSR-2025/
├── sequences_info.json
├── videos/
│   ├── SNGS-001.mp4
│   └── ...
└── images/
    ├── SNGS-001/
    │   └── img1/
    │       ├── 000001.jpg
    │       ├── 000002.jpg
    │       ├── ...
    │       └── 000750.jpg
    └── ...
```

Each video sequence should be extracted into individual frames (`.jpg`), placed under `<sequence_name>/img1/`, and named with zero-padded 6-digit indices starting from `000001.jpg`.

You also need to prepare a `sequences_info.json` file that registers all sequences. The format is as follows:

```json
{
    "version": "1.3",
    "train": [
        {
            "id": 1,
            "name": "SNGS-001",
            "n_frames": 750
        }
    ],
    "validation": [],
    "test": [],
    "challenge": []
}
```

Each entry contains:
- `id`: a unique integer index for the sequence.
- `name`: the directory name of the sequence (must match the folder name under `images/`).
- `n_frames`: the total number of frames in the sequence.

### Pipeline Steps

The data pipeline consists of three steps.

### Step 1: Detection & Tracking

```bash
cd codes/sn-gamestate
CUDA_VISIBLE_DEVICES=0 python -m tracklab.main -cn gsr_step_1_example
```

### Step 2: SAM2 Segmentation Refinement

Use SAM2 to refine the tracking results with precise segmentation masks, then merge the refined results back into the tracker state.

```bash
cd codes/sam2/step_2

# Run SAM2 segmentation inference (supports multi-GPU)
bash gsr_step2_example.sh

# Merge segmentation results back into the tracker state
bash merge_example.sh
```

### Step 3: Remaining Modules

Run the remaining pipeline modules: camera calibration, jersey number recognition, role classification, team assignment, etc.

```bash
cd codes/sn-gamestate
CUDA_VISIBLE_DEVICES=0 python -m tracklab.main -cn gsr_step_3_example_accelerate
```

> **Model selection for Step 3:**
>
> For best results, use `Qwen2.5-VL-72B-Instruct` for jersey number (jn) recognition and `Qwen2.5-VL-7B-Instruct` for role classification (config: `gsr_step_3_example`). However, for faster inference speed and lower GPU memory consumption, you can use the accelerated config (`gsr_step_3_example_accelerate`) which merges both tasks into a single module and uses `Qwen2.5-VL-7B-Instruct` for both, with a minor trade-off in accuracy (Recommended).

## Tactical Analysis and Effects

Runs on top of Stage 1's `predictions.json`, in this order: topology → commentary → audio → effects.

### Stage 2a: GT-first Visible Topology

The topology analyzer uses SoccerNet GameState labels when available and falls back to `outputs/preprocessing/<clip>/predictions.json`. It only describes stable lines formed by visible players; uncertain candidates remain in the JSON and are not rendered.

```bash
python scripts/run_tactical_topology.py analyze --clip-uid SNGS-148
```

The default output is `outputs/tactical_topology/SNGS-148.json`. Each one-second window records the position source, visible-player evidence, accepted lines, and rejected candidates.

### Stage 2b: Two-stage Tactical Commentary

The maintained commentary path first analyzes video evidence with reviewed tactical facts and visible topology, then asks the same provider to write the final commentary. External upload must be explicitly enabled.

```bash
python scripts/run_tactical_commentary.py run \
  --provider doubao \
  --clip-uid soccernetgs:SNGS-148 \
  --allow-external-upload
```

Results are written to `outputs/tactical_commentary/<provider>/<clip>.json`. Topology generation does not add another model call.

### Stage 3: Commentary Audio

Synthesize the commentary with CosyVoice and mux it onto the clip. `--voice default` uses the
built-in voice; `--voice clone` needs a `--prompt-wav` reference sample.

```bash
bash scripts/run_stage3.sh outputs/SNGS-148
```

### Stage 4: Tactical Effects

Stage 4 draws the topology graphics over the frames and muxes the Stage 3 audio into the final
video. Graphics cover the whole clip unless `--window start:end` narrows them; `--focus-track-id`
pins the vertical beam to one track.

```bash
bash scripts/run_stage4.sh outputs/SNGS-148 --clip-dir data/GSR-2025/images/SNGS-148
```

## Multi-Agent Tactic Recognition

Two arms over the frozen 37-clip GSR-2025 set, both runnable on either provider
(`--provider doubao` or `--provider gemini`). They share ground truth, metrics and
reporting, so their numbers are directly comparable.

**A + B + rule → C.** Agent A reports spatial evidence and agent B an event sequence, neither
naming a tactic. Eight deterministic rules score the eight tactics from A and B. Agent C then
arbitrates among the rules' top three candidates against the video. `final.source` records
whether the answer came from the arbiter or the rule.

```bash
python multi-agent/pilot.py run --provider doubao --run-id r5 --allow-external-upload
python multi-agent/pilot.py report --run-id r5
```

**C-only.** The control: a single call classifies the clip directly, with no A, B or rule.

```bash
python multi-agent/c_only.py run --provider gemini --run-id g1 --allow-external-upload
python multi-agent/c_only.py report --run-id g1
```

Both write `multi-agent/outputs/multiagent_pilot/<arm>/<run-id>/` with `clips/`, `manifest.json`,
`metrics.json` and `review.md`. Ground truth is `data/grounded_tactics/gsr2025_gt_37.csv` and the
tactic definitions are `data/足球战术数据库_词条表_Grid.csv`. Uploading video to an external API
requires `--allow-external-upload`.

Offline tests, no network:

```bash
python multi-agent/test_pilot.py
```

## TODO
- [x] Add pretraining code.
- [ ] Refine pretraining code.
- [x] Release SoccerMaster checkpoints.
- [x] Add instructions for quick start.
- [ ] Release datasets.
- [x] Release data pipeline.
