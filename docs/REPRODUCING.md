# Reproducing the experiments

Run every command from the repository root. GPU IDs are passed as a comma-separated list; the examples below use four GPUs.

## Data

The paired loader matches source and target images by relative path. Arrange the datasets as follows:

```text
dataset/
├── QXSLAB_SAROPT/
│   ├── train/{sar,opt}/
│   └── test/{sar,opt}/
├── SpaceNet6_256_clean/
│   ├── train/{sar,opt}/
│   └── test/{sar,opt}/
├── ChesapeakeNAIP_256_clean/
│   ├── train/{nir,rgb}/
│   └── test/{nir,rgb}/
└── Git-10M_resolution_256_1M/
    ├── metadata.jsonl
    └── ...
```

## Checkpoints

The default configs expect this layout:

```text
weights/
├── stage1/ltp_bit_codec_ep19/decoder_ema.pt
├── stage2/igxl_git1m_ep100.pt
├── stage3/
│   ├── qxslab_saropt/igxl_ltp_bit_ep80.pt
│   ├── spacenet6/igxl_ltp_bit_ep80.pt
│   └── chesapeake_naip/igxl_ltp_bit_ep80.pt
└── evaluation/
    ├── inception/weights-inception-2015-12-05-6726825d.pth
    └── clip-vit-large-patch14-336/
```

Paths may be changed in YAML. The launch scripts also accept `CKPT`, `INIT_CKPT`, and `RESUME_CKPT` where noted below.

## Stage 1: codec adaptation

```bash
bash scripts/raev2/stage1/train.sh \
  configs/raev2/stage1/train/codec_adaptation/dinov3l_k7_mix2p5m_ft20.yaml \
  0,1,2,3
```

## Stage 2: target-prior pretraining

Stage 2 is split into an initial 15-epoch run and an 85-epoch continuation.

```bash
bash scripts/raev2/stage2/train.sh \
  configs/raev2/stage2/train/git10m/igxl_1m_ep15.yaml \
  0,1,2,3

INIT_CKPT=weights/stage2/igxl_git1m_ep15.pt \
bash scripts/raev2/stage2/train.sh \
  configs/raev2/stage2/train/git10m/igxl_1m_ep85_from_ep15.yaml \
  0,1,2,3
```

## Stage 3: paired P-DART adaptation

This example trains the QXS-SAROPT model:

```bash
bash scripts/raev2/stage3/train.sh \
  configs/raev2/stage3/train/qxslab_saropt/igxl_s2git1m_ep100_mmdit_encdec_ep80.yaml \
  0,1,2,3
```

SpaceNet6 and Chesapeake have matching configs under `configs/raev2/stage3/train/`. Evaluation runs after training by default. Set `STAGE3_AUTO_EVAL=0` to skip it.

To resume, pass a checkpoint before the same command:

```bash
RESUME_CKPT=/path/to/checkpoint.pt \
bash scripts/raev2/stage3/train.sh <config.yaml> 0,1,2,3
```

## Inference and metrics

```bash
CKPT=weights/stage3/qxslab_saropt/igxl_ltp_bit_ep80.pt \
bash scripts/raev2/stage3/test.sh \
  configs/raev2/stage3/test/qxslab_saropt/test_igxl_s2git1m_ep100_mmdit_encdec_ep80.yaml \
  0,1,2,3
```

The repository includes matching test configs for SpaceNet6 and Chesapeake. Each run writes samples, metrics, and run metadata to the experiment directory named in the config.
