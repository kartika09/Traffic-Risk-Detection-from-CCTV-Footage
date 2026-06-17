# Traffic Risk Detection from CCTV Footage

A two-stage pipeline that detects vehicles in real highway CCTV footage using YOLOv8, engineers traffic-density features from those detections, and classifies each frame into a risk level (`normal`, `congested`, `high_risk`) using XGBoost — with SHAP explainability on top.

![Risk Overlay Demo](assets/demo.gif)

| Normal Traffic | High Risk Traffic |
|---|---|
| ![Normal risk frame](assets/frame_normal.png) | ![High risk frame](assets/frame_high_risk.png) |

The risk label and YOLO detections are overlaid directly on the video frame, with bounding boxes, class labels, and confidence scores shown for every detected vehicle.

## Overview

Most object-detection projects stop at "detect and draw boxes." This project uses YOLOv8 as a **feature extraction layer** rather than an end product: each video frame is converted into a small set of numeric features describing vehicle density, spacing, and composition, which are then fed into a gradient-boosted classifier trained to predict traffic risk.

```
Video (CCTV footage)
        |
        v
YOLOv8m (vehicle detection)
        |
        v
Feature extraction per frame:
  - vehicle_count
  - avg_confidence
  - density_score
  - proximity_score
  - large_vehicle_ratio
        |
        v
XGBoost Classifier --> {normal, congested, high_risk}
        |
        v
SHAP --> feature-level explanation of each prediction
        |
        v
Annotated output video with risk overlay
```

## Why This Approach

YOLO is good at finding objects in pixels; it has no concept of "risk." XGBoost is good at learning patterns in structured numeric data; it cannot interpret raw pixels at all. Rather than training one large end-to-end model (which would require a large labeled video dataset of risk levels and significant compute), this pipeline reuses YOLO's pretrained detection capability and layers a lightweight, fully interpretable classifier on top of the features it produces.

## Dataset & Labeling

Frames were sampled every 5th frame from real highway CCTV footage (~10,000 data points after sampling). Risk labels were derived using **percentile-based thresholds** on vehicle count (33rd/66th percentile splits), rather than fixed hand-picked numbers. An earlier version used fixed thresholds (e.g. "5+ vehicles = high risk"), which produced a severely imbalanced dataset (10,202 `high_risk` vs. 39 `normal` frames) because this footage is a consistently busy motorway. Switching to data-driven percentile thresholds fixed this and produced a balanced split across all three classes.

## A Note on Label Leakage

An earlier version of this model scored a perfect 1.00 across precision, recall, and F1 — which is a red flag, not a success. The cause: `vehicle_count` was used both to construct the labels and as a training feature, so the model was simply learning the labeling rule rather than a generalizable pattern. Removing `vehicle_count` from the feature set dropped accuracy to a realistic **94%**, which is the number reported below. This is a deliberate methodological fix, not an oversight — the model's remaining features (density, proximity, confidence, vehicle mix) still carry strong signal even without directly counting vehicles.

## Results

**Accuracy: 94%**, balanced across all three classes.

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| congested | 0.92 | 0.92 | 0.92 |
| high_risk | 0.94 | 0.94 | 0.94 |
| normal | 0.94 | 0.95 | 0.94 |

![Confusion Matrix](assets/confusion_matrix.png)

Most misclassifications happen at the boundary between adjacent classes (e.g. `congested` predicted as `high_risk`, or `normal` predicted as `congested`) rather than between distant classes — `normal` is essentially never confused with `high_risk`. This is expected: traffic density is a continuum, and frames near a percentile boundary are inherently the hardest to classify correctly.

## Explainability (SHAP)

`density_score` (proportion of the frame covered by vehicle bounding boxes) emerged as the most influential feature across all three risk classes, with `avg_confidence` and `proximity_score` contributing secondary signal. This shows that spatial coverage and spacing patterns alone — without directly counting vehicles — carry most of the predictive signal for traffic risk.

![SHAP Summary](assets/shap_summary.png)

## Tech Stack

Python, Google Colab, YOLOv8 (Ultralytics), OpenCV, XGBoost, scikit-learn, SHAP, pandas/numpy.

## Pipeline Details

**Detection:** `yolov8m.pt`, filtered to vehicle classes (car, motorcycle, bus, truck), confidence threshold 0.25. The medium variant was chosen after the nano variant failed to reliably detect vehicles in this footage (1 false detection at 0.29 confidence vs. 14 correct detections at up to 0.94 confidence with the medium model).

**Features (per sampled frame):**
- `vehicle_count` — total detected vehicles (used for labeling only, excluded from training features)
- `avg_confidence` — mean detection confidence; drops when vehicles overlap/occlude each other
- `density_score` — total vehicle bounding-box area ÷ frame area
- `proximity_score` — inverse of mean pairwise distance between vehicle centers
- `large_vehicle_ratio` — proportion of detections that are buses/trucks vs. cars/motorcycles

**Classifier:** XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`), 80/20 stratified train/test split.

**Explainability:** SHAP `TreeExplainer`, exact Shapley values for the trained XGBoost model.

## Limitations

Risk labels are a constructed proxy based on vehicle density percentiles within this specific footage, not verified real-world accident/incident data — this pipeline demonstrates that visual congestion features can predict a density-based risk proxy, not a validated accident-prediction system. The model has only been trained and evaluated on a single camera angle and road type; generalization to other viewpoints, weather conditions, or road types is untested.

## Project Structure

```
traffic-risk-detection/
├── notebook.ipynb
├── README.md
└── assets/
    ├── confusion_matrix.png
    ├── shap_summary.png
    ├── frame_normal.png
    ├── frame_high_risk.png
    └── demo.gif
```

## Possible Extensions

Incorporating temporal features (rate of change in density across consecutive frames, not just single-frame snapshots) since real risk often comes from sudden changes rather than steady-state congestion; validating against human-annotated or incident-based ground truth instead of a density-derived proxy; testing generalization on footage from different camera angles and road types.
