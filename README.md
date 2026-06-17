Traffic Risk Detection from CCTV Footage

A two-stage pipeline that detects vehicles in real highway CCTV footage using YOLOv8, extracts traffic-density features from the detections, and classifies each frame's risk level (normal, congested, high_risk) using XGBoost, with SHAP for explainability.

How It Works


Sample frames from CCTV video footage
Detect vehicles per frame using YOLOv8m
Extract 5 features: vehicle count, average detection confidence, density score, proximity score, large-vehicle ratio
Label frames as normal/congested/high_risk using percentile-based thresholds
Train an XGBoost classifier on the features (excluding vehicle count, to avoid label leakage)
Use SHAP to explain which features drive each prediction
Overlay the predicted risk level back onto the video


Results

94% accuracy, balanced across all three classes.

ClassPrecisionRecallF1-scorecongested0.920.920.92high_risk0.940.940.94normal0.940.950.94

SHAP Explainability

density_score was the most influential feature across all three classes, followed by avg_confidence and proximity_score.

A Note on Label Leakage

An earlier version of this model scored a perfect 1.00 across all metrics — a sign something was wrong, not a good result. The cause: vehicle count was used to create the labels but was also included as a training feature, so the model was just learning the labeling rule. Removing it from the feature set dropped accuracy to a realistic 94%, which is the result reported above.

Tech Stack

Python, Google Colab, YOLOv8 (Ultralytics), OpenCV, XGBoost, scikit-learn, SHAP, pandas/numpy

Project Structure

traffic-risk-detection/
├── notebook.ipynb
└── README.md

Limitations

Risk labels are a density-based proxy derived from this specific footage, not verified real-world accident data. The model has only been tested on a single camera angle and road type.
