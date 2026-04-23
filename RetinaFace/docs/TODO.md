# Face Localization & Classification Project

## Pipeline Structure

1. Input image
2. Face detector -> bounding boxes (`bbox`)  (owner: me)
3. Face localizer -> facial landmarks
4. Face classifier -> age, gender

## TODO (Face Detector)

### 1. Reduce dataset volume safely
- Subsample WIDER FACE to fit the project scope and training budget.
- Keep class and difficulty balance (easy/medium/hard) to avoid distribution shift.
- Define and document the sampling rule for reproducibility.

Done when:
- A reduced dataset manifest is created and versioned.
- The split keeps comparable difficulty distribution to the original set.

### 2. Real-time bounding box visualization
- Build a real-time detection visualization tool with OpenCV.
- Display bounding boxes, confidence score, and FPS on each frame.

Done when:
- Webcam or video input runs end-to-end with overlayed detections.
- Visualization can be started from a single command.

### 3. Data augmentation with ground truth consistency
- Add augmentation while keeping bbox/landmark labels synchronized with transformed images.
- Verify transforms such as flip, resize, crop, and color jitter.

Done when:
- Augmented samples pass sanity-check visualization.
- No invalid bbox/landmark labels are generated.

### 4. Find and validate a pretrained checkpoint
- Select a pretrained backbone/checkpoint (e.g., MobileNet0.25 or ResNet50).
- Verify compatibility with the current training/inference code path.

Done when:
- Checkpoint loads without key mismatch issues.
- Baseline evaluation runs successfully with logged metrics.

