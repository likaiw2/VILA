# ReVA VILA1.5-3B Local Instructions

This note documents the local workflow that has been validated in this repository for:

- inference with `Efficient-Large-Model/VILA1.5-3b`
- evaluation on `ReVA_V1` and `RSVidQA`
- LoRA fine-tuning on `.data/ReVA_V1/final_train_set.json`

It is intentionally narrow. The commands below reflect the current local repo state and the fixes already applied in this checkout.

## 1. Environment Installation

### Recommended path

Create a clean conda env first:

```bash
conda create -n vila python=3.10.14 -y
conda activate vila
cd /home/liw324/code/VILA
```

Upgrade packaging tools:

```bash
pip install --upgrade pip setuptools
```

Install FlashAttention 2 for the local torch / CUDA stack expected by this repo:

```bash
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Install VILA itself:

```bash
pip install -e ".[train,eval]"
```

Install the Triton version used by this local workflow:

```bash
pip install triton==2.3.0
```

Patch the local DeepSpeed package with the repo replacement files:

```bash
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* "$site_pkg_path/deepspeed/"
```

Downgrade protobuf for tokenizer compatibility:

```bash
pip install protobuf==3.20.3
```

### One-command installer

You can also use the repo helper script:

```bash
bash environment_setup.sh vila
```

or, if you already created and activated the env yourself:

```bash
conda activate vila
bash environment_setup.sh
```

### Important local install notes

The current local training and inference path assumes the following:

- In our program, we don't use `ps3-torch`. So, do not install it. It conflicts with `vila==2.0.0` on `timm`.
- If `environment_setup.sh` or `pip install -e ".[train,eval]"` pulls `ps3-torch@git+https://github.com/NVlabs/PS3`, remove that dependency from the install path for this workflow.
- If `wandb` conflicts with `protobuf==3.20.3`, the simplest local workaround is to remove `wandb` or use `REPORT_TO=none`.
- If you see a protobuf compatibility error during tokenizer loading, verify:

```bash
python -c "import google.protobuf; print(google.protobuf.__version__)"
```

It should print `3.20.3`.

### Why these deviations are needed

These were the actual local issues encountered during setup:

- `ps3-torch` required `timm==1.0.15`
- `vila==2.0.0` required `timm==0.9.12`
- protobuf versions newer than `3.20.x` broke the local tokenizer / sentencepiece path

This document follows the path that was actually validated for `Efficient-Large-Model/VILA1.5-3b`.

## 2. Minimal End-to-End Checklist

If you only want the shortest working path, run this sequence:

```bash
conda activate vila
cd /home/liw324/code/VILA

python data_prepare/reva/convert_reva_to_vila.py \
  --input .data/ReVA_V1/final_train_set.json \
  --output .data/ReVA_V1/reva_v1_train_vila.json

python scripts/eval/reva.py \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --question-file .data/ReVA_V1/final_valid_set.json \
  --dataset-root .data \
  --output-dir runs/eval/reva-valid-smoke \
  --max-questions 10

VILA_DATASETS=reva_v1 \
GPUS=1 \
GLOBAL_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=8 \
NUM_VIDEO_FRAMES=4 \
MODEL_MAX_LENGTH=2048 \
REPORT_TO=none \
bash scripts/NVILA-Lite/sft_local_lora_only.sh \
  Efficient-Large-Model/VILA1.5-3b \
  reva_v1_train \
  runs/train/reva-v1-vila15-3b-lora
```

## 3. Environment Usage

Use the `vila` conda environment:

```bash
conda activate vila
cd /home/liw324/code/VILA
```

Local assumptions used by the commands below:

- `.data` points to the dataset root by symlink
- `Efficient-Large-Model/VILA1.5-3b` is accessible from the current environment
- training logs are written to `runs/log/<timestamp>.log`

Important local notes:

- Do not install `ps3-torch` for this workflow. It conflicts with the `timm` version required by `vila`.
- If `wandb` causes a `protobuf` conflict, the simplest local path is to disable remote reporting with `REPORT_TO=none`.
- This repo copy already contains local fixes for:
  - optional FP8 import fallback
  - VILA 1.5 chat template compatibility during training
  - tokenizer / embedding resize during training
  - SigLIP patch-token handling in the vision tower
  - ReVA path normalization during dataset conversion

## 4. Quick Inference

Single video inference:

```bash
python -m llava.cli.infer \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --media .data/RSVidQA/DJI_Split/split_NJ/DJI_0157_d4_01.mp4 \
  --text "What is happening in this video? Answer briefly." \
  --num_video_frames 4
```

LoRA checkpoint inference after fine-tuning:

```bash
python -m llava.cli.infer \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --lora-path runs/train/reva-v1-vila15-3b-lora/model/checkpoint-200 \
  --media .data/RSVidQA/DJI_Split/split_NJ/DJI_0157_d4_01.mp4 \
  --text "What is happening in this video? Answer briefly." \
  --num_video_frames 4
```

## 5. ReVA Validation Evaluation

Run evaluation on the ReVA validation split:

```bash
python scripts/eval/reva.py \
  --model-path Efficient-Large-Model/VILA1.5-3b \
  --question-file .data/ReVA_V1/final_valid_set.json \
  --dataset-root .data \
  --output-dir runs/eval/reva-valid
```

Useful optional flags:

- `--max-questions 10` for a smoke test
- `--resume` to continue an interrupted run
- `--num-video-frames 4` to override the model config

Outputs:

- `runs/eval/reva-valid/outputs.jsonl`
- `runs/eval/reva-valid/metrics.json`

## 6. Convert ReVA Training Data

The raw training file is not in VILA training format. Convert it first:

```bash
python data_prepare/reva/convert_reva_to_vila.py \
  --input .data/ReVA_V1/final_train_set.json \
  --output .data/ReVA_V1/reva_v1_train_vila.json
```

This converter:

- reads `data["questions"]`
- converts each item to `LLaVADataset` format
- normalizes broken absolute paths
- strips the `#dataset` prefix when present
- fixes `SRA_Select/` to `ERA_Select/` when needed

The registered dataset entry is already present in:

- [reva_v1.yaml](/home/liw324/code/VILA/llava/data/registry/datasets/reva_v1.yaml)

It currently points to:

```yaml
reva_v1_train:
  _target_: llava.data.LLaVADataset
  data_path: .data/ReVA_V1/reva_v1_train_vila.json
  media_dir: .data/ReVA_V1
  is_video: true
```

## 7. Launch LoRA Fine-Tuning

This is the validated local command for `VILA1.5-3b`:

```bash
VILA_DATASETS=reva_v1 \
GPUS=1 \
GLOBAL_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=8 \
NUM_VIDEO_FRAMES=4 \
MODEL_MAX_LENGTH=2048 \
REPORT_TO=none \
bash scripts/NVILA-Lite/sft_local_lora_only.sh \
  Efficient-Large-Model/VILA1.5-3b \
  reva_v1_train \
  runs/train/reva-v1-vila15-3b-lora
```

Notes:

- On 1 GPU, this script resolves to `per_device_train_batch_size=1`.
- The script writes a log file to `runs/log/<timestamp>.log`.
- Checkpoints are saved under `runs/train/reva-v1-vila15-3b-lora/model`.

If memory is tight, reduce context or frames:

```bash
VILA_DATASETS=reva_v1 \
GPUS=1 \
GLOBAL_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=8 \
NUM_VIDEO_FRAMES=2 \
MODEL_MAX_LENGTH=1024 \
REPORT_TO=none \
bash scripts/NVILA-Lite/sft_local_lora_only.sh \
  Efficient-Large-Model/VILA1.5-3b \
  reva_v1_train \
  runs/train/reva-v1-vila15-3b-lora
```

## 8. How To Tell Training Is Healthy

Healthy training looks like this in `runs/log/<timestamp>.log`:

- no `Traceback`
- no `RuntimeError`
- dataloader created successfully, for example `length of dataloader: 10072 10072`
- progress starts moving, for example `1/1259`, `2/1259`
- loss lines appear, for example:

```text
{'loss': 0.3584, 'grad_norm': 2.1103..., 'learning_rate': ..., 'epoch': 0.0}
```

This means:

- model load succeeded
- forward pass succeeded
- backward pass succeeded
- optimizer step succeeded

Warning lines like the following are not fatal by themselves:

- `Could not estimate the number of tokens of the input`
- `c10d::broadcast_ ... autograd kernel was not registered`

## 9. Common Failure Modes

### `ModuleNotFoundError: FloatPointQuantizeTorch`

Cause:

- FP8 / quantization modules are unavailable locally

Handling:

- use the current repo copy, which already makes those imports optional for the normal VILA 1.5 path
- do not enable quantized training for this workflow

### `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`

Cause:

- legacy VILA 1.5 tokenizer path during training

Handling:

- fixed in this repo copy by setting conversation mode before tokenizer construction

### `indexSelectLargeIndex` or CUDA device-side assert in the first step

Cause:

- tokenizer size and embedding size out of sync after media tokens were added

Handling:

- fixed in this repo copy by resizing token embeddings during training startup

### `shape '[2, 26, 26, -1]' is invalid`

Cause:

- SigLIP patch tokens were incorrectly treated like CLIP patch+CLS output

Handling:

- fixed in this repo copy by preserving all patch tokens for SigLIP

### `Video '...' has no frames`

Cause:

- bad training path in converted annotations

Handling:

- regenerate the converted ReVA training file with the converter in `data_prepare/reva/convert_reva_to_vila.py`

## 10. Useful Paths

- training script: [sft_local_lora_only.sh](/home/liw324/code/VILA/scripts/NVILA-Lite/sft_local_lora_only.sh)
- ReVA converter: [convert_reva_to_vila.py](/home/liw324/code/VILA/data_prepare/reva/convert_reva_to_vila.py)
- ReVA eval script: [reva.py](/home/liw324/code/VILA/scripts/eval/reva.py)
- dataset registry: [reva_v1.yaml](/home/liw324/code/VILA/llava/data/registry/datasets/reva_v1.yaml)
