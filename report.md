# CS5330 Project 5: Recognition using Deep Networks
**Name:** [Your Name]
**Course:** CS5330 – Pattern Recognition and Computer Vision
**Date:** Spring 2026

---

## 1. Project Overview

This project explores building, training, analyzing, and modifying deep convolutional neural networks for digit recognition using the MNIST dataset. The project uses the PyTorch framework and covers five main tasks: training a CNN from scratch, examining its learned filters, applying transfer learning to recognize Greek letters, replacing the CNN with a Vision Transformer architecture, and running a systematic hyperparameter experiment to optimize performance. The MNIST dataset consists of 60,000 training and 10,000 test 28×28 grayscale digit images, making it a compact but representative benchmark for evaluating deep learning techniques.

---

## 2. Task Results and Images

### Task 1A – First 6 Test Set Digits

![First 6 MNIST test examples](test_examples.png)

The first six examples of the MNIST test set, shown without shuffling so the same digits appear each run. Labels are shown above each image.

---

### Task 1B – Network Architecture

The CNN has the following layer structure:

| Layer | Details |
|-------|---------|
| Conv2d | 1 → 10 channels, 5×5 kernel |
| MaxPool2d + ReLU | 2×2 window |
| Conv2d | 10 → 20 channels, 5×5 kernel |
| Dropout | 50% rate |
| MaxPool2d + ReLU | 2×2 window |
| Flatten | 20×4×4 = 320 features |
| Linear + ReLU | 320 → 50 |
| Linear + log_softmax | 50 → 10 |

*[Include your hand-drawn or generated network diagram here]*

---

### Task 1C – Training and Test Accuracy

![Training and test accuracy curves](training_curves.png)

Training and test accuracy plotted per epoch over 5 epochs. The network reaches approximately [X]% test accuracy after 5 epochs.

---

### Task 1E – Test Set Evaluation

The table below shows the network output for the first 10 test examples:

| Idx | Network Output (10 values) | Predicted | Correct |
|-----|---------------------------|-----------|---------|
| 0   | [paste output here]        | [pred]    | [label] |
| 1   | ...                        |           |         |
| 2   | ...                        |           |         |
| 3   | ...                        |           |         |
| 4   | ...                        |           |         |
| 5   | ...                        |           |         |
| 6   | ...                        |           |         |
| 7   | ...                        |           |         |
| 8   | ...                        |           |         |
| 9   | ...                        |           |         |

![First 9 test predictions](test_predictions.png)

The 3×3 grid shows the first 9 test images with predicted and true labels. Green titles indicate correct predictions, red indicate errors.

---

### Task 1F – Handwritten Digit Test

![Handwritten digit results](handwritten_results.png)

Results of running my own handwritten digits through the trained network. Images were resized to 28×28 and inverted to match the MNIST white-on-black format. [Describe which digits were correctly/incorrectly classified and any observations.]

---

### Task 2A – conv1 Filter Visualization

![conv1 filters](conv1_filters.png)

The 10 learned 5×5 filters from the first convolutional layer, displayed in a 3×4 grid. [Describe what patterns you see – e.g., edge detectors, blob detectors, etc.]

---

### Task 2B – Filter Effects on Training Image

![conv1 filter effects](conv1_filter_effects.png)

The effect of applying each of the 10 conv1 filters to the first training image using `cv2.filter2D`. [Note whether the results make sense given the filter shapes. For example, filters resembling horizontal/vertical edge detectors should highlight those edges in the output.]

---

### Task 3 – Transfer Learning on Greek Letters

**Modified network (fc2 replaced with 3-node output layer):**

```
[Paste model printout here]
```

**How many epochs to near-perfect classification:** [X] epochs

![Greek letter training loss](greek_training_loss.png)

The training loss decreases rapidly, reflecting the small dataset (27 examples) and frozen pretrained features.

![Personal Greek letter results](personal_greek_results.png)

Results on my own handwritten alpha, beta, and gamma symbols. [Describe accuracy and any misclassifications.]

---

### Task 4 – Transformer Network Results

**Config used:** Default `NetConfig` (patch_size=4, stride=2, embed_dim=48, depth=4, heads=8, epochs=15)

![Transformer accuracy](transformer_accuracy.png)

The transformer reached approximately [X]% test accuracy after 15 epochs. [Compare to the CNN result and note any differences in convergence speed or final accuracy.]

---

### Task 5 – Architecture Experiment

**Three dimensions evaluated:**

| Dimension | Values tested |
|-----------|--------------|
| Conv1 filter count | 5, 10, 20, 40 |
| FC1 hidden nodes | 25, 50, 100, 200 |
| Dropout rate | 0.1, 0.3, 0.5, 0.7 |

**Hypotheses (stated before running):**
- Conv1 filters: More filters should capture richer low-level features, improving accuracy.
- FC1 nodes: Larger hidden layer should help up to a point; beyond that, gains diminish.
- Dropout: Moderate dropout (0.3–0.5) should generalize best; extremes may underfit or overfit.

**Results (64 total configurations, 3 epochs each):**

![Experiment results](experiment_plots.png)

*Top configuration:* conv1=[X], fc1=[X], dropout=[X] → [X]% accuracy
*Bottom configuration:* conv1=[X], fc1=[X], dropout=[X] → [X]% accuracy

**Discussion:** [Did the results match your hypotheses? Note any surprising findings.]

---

## 3. Extensions

*[Describe any extensions attempted, e.g., more Greek letters, Gabor filter replacement, live digit recognition, etc.]*

---

## 4. Reflection

*[Write 2–4 sentences about what you learned from this project. What surprised you? What was the most challenging part?]*

---

## 5. Acknowledgements

- PyTorch documentation and tutorials: https://pytorch.org/tutorials/
- MNIST dataset: Yann LeCun et al.
- Assignment specification provided by the course instructor.
- *[List any other people or resources consulted.]*
