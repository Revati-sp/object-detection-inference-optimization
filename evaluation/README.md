# Evaluation of Object Detection Inference Pipeline

## 1. Overview

This project evaluates the performance of object detection models on real-world video data. The evaluation focuses on two key aspects:

* **Accuracy** (mAP) using annotated image data
* **Speed** (latency and FPS) using video inference

The goal is to compare multiple models and inference backends to understand the trade-off between accuracy and performance.

---

## 2. Dataset Description

A custom dataset was created using real videos recorded by the user.

* **Number of videos**: 2
* **Total extracted images**: 139
* **Total annotations**: 374 bounding boxes
* **Number of classes**: 36
* **Resolution**: ~576 × 1024

Images were extracted from videos and annotated in **COCO format**.
The annotation file used:

data/annotations/instances_custom.json

This ensures that accuracy (mAP) is evaluated on **user-generated data with custom annotations**, as required.

---

## 3. Models and Backends

### Models

* YOLOv8 (Ultralytics)
* YOLOv5 (torch.hub)

### Inference Backends

* PyTorch (baseline)
* TorchScript (compiled)
* ONNX Runtime (optimized)

---

## 4. Accuracy Evaluation (mAP)

Accuracy is measured using **Mean Average Precision (mAP)**:

* **mAP@0.5**
* **mAP@0.5:0.95**

Evaluation is performed using:

scripts/evaluate_dataset.py

Results are stored in:

results/eval_report.csv

This evaluation uses the custom annotated dataset, ensuring that accuracy reflects real-world performance.

---

## 5. Speed Evaluation (Latency & FPS)

Performance is measured using:

* **Latency per frame (ms)**
* **Frames per second (FPS)**

Benchmarking is performed using:

scripts/benchmark_models.py

Results are stored in:

results/benchmark.csv

---

## 6. Video Inference Evaluation

Video-based inference is performed using real videos from:

data/videos/

Metrics collected:

* Total frames processed
* Average FPS
* Average latency per frame
* Total detections

Example observations:

* YOLOv8 (PyTorch): ~37 FPS, ~26 ms latency
* YOLOv5 (PyTorch): ~25 FPS, ~39 ms latency

Video benchmark results:

results/video_benchmark.csv

Annotated output videos are saved in:

backend/outputs/

---

## 7. Frontend Visualization

A Next.js frontend is used to:

* Upload an image or video file
* Select a model (YOLOv8 / YOLOv5) and inference backend (PyTorch / TorchScript / ONNX)
* Run inference for both **image** and **video** workflows
* Visualize bounding boxes with class labels and confidence scores
* Display latency (ms) and FPS metrics for every request

Both workflows have been verified end-to-end:

* **Image inference UI** — upload a JPEG/PNG, receive annotated bounding boxes and per-request latency
* **Video inference UI** — upload an MP4, receive frame-level detections, average FPS, and total detections
* **Output visibility** — bounding boxes, confidence scores, latency (ms), and FPS are all shown in the frontend after each run

---

### Image Evaluation Screenshots

Screenshots captured from the image inference workflow (upload → model selection → detection results):

![Image eval 1](Image%20evaluation/Screenshot%202026-04-17%20at%2010.26.51%E2%80%AFPM.png)
![Image eval 2](Image%20evaluation/Screenshot%202026-04-17%20at%2010.27.06%E2%80%AFPM.png)
![Image eval 3](Image%20evaluation/Screenshot%202026-04-17%20at%2010.27.16%E2%80%AFPM.png)
![Image eval 4](Image%20evaluation/Screenshot%202026-04-17%20at%2010.28.05%E2%80%AFPM.png)
![Image eval 5](Image%20evaluation/Screenshot%202026-04-17%20at%2010.29.41%E2%80%AFPM.png)
![Image eval 6](Image%20evaluation/Screenshot%202026-04-17%20at%2010.37.18%E2%80%AFPM.png)
![Image eval 7](Image%20evaluation/Screenshot%202026-04-17%20at%2010.37.26%E2%80%AFPM.png)
![Image eval 8](Image%20evaluation/Screenshot%202026-04-17%20at%2010.37.31%E2%80%AFPM.png)
![Image eval 9](Image%20evaluation/Screenshot%202026-04-17%20at%2010.41.44%E2%80%AFPM.png)
![Image eval 10](Image%20evaluation/Screenshot%202026-04-17%20at%2010.41.50%E2%80%AFPM.png)
![Image eval 11](Image%20evaluation/Screenshot%202026-04-17%20at%2010.42.09%E2%80%AFPM.png)
![Image eval 12](Image%20evaluation/Screenshot%202026-04-17%20at%2010.42.13%E2%80%AFPM.png)
![Image eval 13](Image%20evaluation/Screenshot%202026-04-17%20at%2010.42.41%E2%80%AFPM.png)
![Image eval 14](Image%20evaluation/Screenshot%202026-04-17%20at%2010.42.44%E2%80%AFPM.png)
![Image eval 15](Image%20evaluation/Screenshot%202026-04-17%20at%2010.43.00%E2%80%AFPM.png)
![Image eval 16](Image%20evaluation/Screenshot%202026-04-17%20at%2010.43.01%E2%80%AFPM.png)
![Image eval 17](Image%20evaluation/Screenshot%202026-04-17%20at%2010.43.17%E2%80%AFPM.png)
![Image eval 18](Image%20evaluation/Screenshot%202026-04-17%20at%2010.43.18%E2%80%AFPM.png)

---

### Video Evaluation Screenshots

Screenshots captured from the video inference workflow (upload → model selection → FPS, latency, total detections):

![Video eval 1](video%20evaluation/Screenshot%202026-04-17%20at%2010.44.36%E2%80%AFPM.png)
![Video eval 2](video%20evaluation/Screenshot%202026-04-17%20at%2010.45.14%E2%80%AFPM.png)
![Video eval 3](video%20evaluation/Screenshot%202026-04-17%20at%2010.46.10%E2%80%AFPM.png)
![Video eval 4](video%20evaluation/Screenshot%202026-04-17%20at%2010.46.49%E2%80%AFPM.png)
![Video eval 5](video%20evaluation/Screenshot%202026-04-17%20at%2010.47.43%E2%80%AFPM.png)
![Video eval 6](video%20evaluation/Screenshot%202026-04-17%20at%2010.49.27%E2%80%AFPM.png)


---

## 8. Performance Analysis

### Model Comparison

* YOLOv8 achieves higher speed and more stable latency
* YOLOv5 is slower but still reliable

### Backend Comparison

* PyTorch provides stable baseline performance
* TorchScript introduces overhead in this setup
* ONNX Runtime provides competitive or improved speed depending on configuration

### Trade-off

* Faster inference (higher FPS) may slightly affect accuracy
* Optimal choice depends on deployment scenario:

  * Real-time systems → prefer faster backend (YOLOv8 + ONNX/PyTorch)
  * Accuracy-critical tasks → prefer higher mAP configurations

---

## 9. Conclusion

This project demonstrates a complete inference optimization pipeline:

* Real-world dataset with custom annotations
* Multi-model and multi-backend evaluation
* Quantitative comparison of accuracy and performance
* End-to-end deployment with backend and frontend

The results show clear trade-offs between speed and accuracy, providing insights into selecting the appropriate model and backend for real-time applications.
