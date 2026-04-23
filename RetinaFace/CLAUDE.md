# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a PyTorch implementation of RetinaFace for single-stage face detection. It is the **face detector** component (step 2) of a larger pipeline: image → face detector (bounding boxes) → face localizer (landmarks) → face classifier (age/gender).

The implementation lives entirely under `Pytorch_Retinaface/`. All commands below should be run from that directory.

## Commands

**Train:**
```bash
# MobileNet0.25 backbone (lightweight, single GPU)
python train.py --network mobile0.25

# ResNet50 backbone (higher accuracy, multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50

# Resume training from checkpoint
python train.py --network mobile0.25 --resume_net ./weights/mobilenet0.25_epoch_50.pth --resume_epoch 50
```

**Inference / detect on a single image:**
```bash
python detect.py -m ./weights/Resnet50_Final.pth --network resnet50
python detect.py -m ./weights/mobilenet0.25_Final.pth --network mobile0.25 --cpu
```

**Evaluate on WIDER FACE val:**
```bash
# 1. Generate prediction txt files
python test_widerface.py --trained_model ./weights/Resnet50_Final.pth --network resnet50

# 2. Build Cython extension (one-time)
cd widerface_evaluate && python setup.py build_ext --inplace

# 3. Run evaluation
python widerface_evaluate/evaluation.py
```

**Evaluate on FDDB:**
```bash
python test_fddb.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25
```

**Export to ONNX:**
```bash
python convert_to_onnx.py
```

## Architecture

**Model (`models/retinaface.py`):**
- Backbone (MobileNetV1 or ResNet50) extracts feature maps at 3 scales via `IntermediateLayerGetter`
- FPN (`models/net.py:FPN`) fuses multi-scale features
- Three SSH (Single Stage Headless) modules (`models/net.py:SSH`) process each FPN level
- Three parallel prediction heads per FPN level: `BboxHead` (4 coords), `ClassHead` (2-class), `LandmarkHead` (10 coords for 5 landmarks)
- In train mode returns raw logits; in test mode applies softmax to classification output

**Loss (`layers/modules/multibox_loss.py`):**
- Combined: `loc_weight * smooth_L1_loc + cross_entropy_cls + smooth_L1_landmark`
- Hard negative mining with 7:1 neg:pos ratio
- `loc_weight = 2.0` (set in `data/config.py`)

**Anchors (`layers/functions/prior_box.py`):**
- 3 FPN levels with steps `[8, 16, 32]`
- Min sizes: `[[16,32], [64,128], [256,512]]`
- Generated once before training and cached on GPU

**Data pipeline (`data/wider_face.py`, `data/data_augment.py`):**
- Reads WIDER FACE annotations from `label.txt`
- Target tensor format per annotation: `[x1, y1, x2, y2, lm0x, lm0y, ..., lm4x, lm4y, label]` → 15 values

## Configuration

All hyperparameters are in `data/config.py` as two dicts — `cfg_mnet` (MobileNet0.25) and `cfg_re50` (ResNet50). Key fields: `batch_size`, `epoch`, `decay1/decay2` (LR step epochs), `image_size`, `loc_weight`, `ngpu`.

## Weights Layout

```
./weights/
    mobilenetV1X0.25_pretrain.tar    # backbone pretrain (required before training MobileNet)
    mobilenet0.25_Final.pth          # trained model
    Resnet50_Final.pth               # trained model
```

The MobileNet pretrain is loaded automatically during `RetinaFace.__init__` when `cfg['pretrain'] = True`.

## Dataset Layout

```
./data/widerface/
    train/
        images/
        label.txt
    val/
        images/
        wider_val.txt       # filenames only, no labels
./data/FDDB/
    images/
    img_list.txt
```

## Active Work (docs/TODO.md)

1. **Dataset reduction** — stratified subsampling of WIDER FACE preserving easy/medium/hard distribution. Methodology documented in `docs/dataset-reduction-methodology.md`. Artifacts: `subset_v1_manifest.csv`, `subset_v2_manifest.csv`, `sampling_config.yaml`.
2. **Real-time visualization** — OpenCV-based webcam/video detection with bounding boxes, confidence scores, and FPS overlay.
3. **Data augmentation** — verify flip/resize/crop/color-jitter keep bbox and landmark labels consistent.
4. **Pretrained checkpoint validation** — ensure selected checkpoint loads without key mismatch.
