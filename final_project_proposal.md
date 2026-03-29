# CS5330 Final Project Proposal
## Real-Time Object Recognition and Augmented Reality Fusion

**Team Members:**
Sangeeth Deleep Menon — NUID 002524579 — MSCS Boston — Section 03 (CRN 40669, Online)
Raj Gupta — NUID 002068701 — MSCS Boston — Section 01 (CRN 38745, Online)

---

### Project Overview

We propose building a **real-time object recognition and augmented reality system** that fuses the deep learning pipeline from Project 5 with the AR rendering engine from Project 4. The system will detect and classify objects from a webcam feed using a trained convolutional neural network, then overlay contextual AR annotations — labels, bounding boxes, and simple 3D graphics — directly onto each recognized object without requiring a printed chessboard or ArUco marker.

This extends both prior projects in a meaningful way: Project 4 required a fixed physical target for pose estimation, and Project 5's recognition ran offline on still images. Combining them produces a system where **any object the network can recognize becomes an AR anchor**.

---

### Planned Extensions and Scope

**1. Multi-class CNN recognizer (extension of Project 5)**
We will train or fine-tune a CNN on a custom dataset of 5–10 everyday object classes (e.g., cup, phone, book, keyboard, pen). We will compare a standard CNN architecture against a Vision Transformer to select the better backbone, building directly on our Task 4 and Task 5 work.

**2. Markerless AR with deep-learning pose estimation (extension of Project 4)**
Rather than relying on a chessboard, we will use the bounding box from the classifier together with a depth estimate (from monocular cues or a fixed camera model) to compute a rough pose and render a 3D label tag or simple 3D object anchored to each recognized item.

**3. Real-time pipeline and GUI**
The system will run at interactive frame rates on a standard laptop using PyTorch with MPS/CUDA acceleration. A tkinter or OpenCV GUI will display the live feed, the detected class, confidence, and AR overlays simultaneously — combining the GUI work done in both projects.

**4. Quantitative evaluation**
We will report per-class precision and recall on a held-out test set, and measure end-to-end latency (inference + rendering) to demonstrate real-time performance.

---

### Division of Work

| Component | Owner |
|-----------|-------|
| Dataset collection and CNN training | Sangeeth |
| AR overlay rendering and pose pipeline | Raj |
| GUI and integration | Both |
| Evaluation and report | Both |

---

### Why This Project

This project is scoped appropriately for a two-person team: it requires non-trivial extensions to two separate prior projects, produces a visually compelling real-time demo, and has clear, measurable deliverables. It is larger in scope than either Project 4 or Project 5 individually, combining computer vision, deep learning, and AR in a single unified system.
