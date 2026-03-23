# CS5330 Project 5 – Project Guide

## Status Overview

| Task | File | Status |
|------|------|--------|
| Task 1A–D: Build & train CNN | `task1.py` | Code complete, needs to be run |
| Task 1E–F: Evaluate on test set & handwritten digits | `task1_test.py` | Code complete, needs to be run |
| Task 2A–B: Examine filters | `task2.py` | Code complete, needs to be run |
| Task 3: Greek letter transfer learning | `task3.py` | Code complete, needs to be run |
| Task 4: Transformer network | `task4.py` + `NetTransformer-template.py` | Code complete, needs to be run |
| Task 5: Architecture experiment | `task5.py` | Code complete, needs to be run |
| Report | `report.md` | Template ready, needs results filled in |
| Submission | — | Not yet done |

---

## What Has Been Done

### Code
- **`task1.py`** — Builds and trains the MNIST CNN (5 epochs, SGD lr=0.01, momentum=0.5, batch=64). Saves model to `mnist_model.pth`. Plots accuracy curves to `training_curves.png`.
- **`task1_test.py`** — Loads `mnist_model.pth`, evaluates on first 10 test examples, plots predictions to `test_predictions.png`. Also tests on personal handwritten digit images from `handwritten_digits/`.
- **`task2.py`** — Loads trained model, visualizes 10 conv1 filters to `conv1_filters.png`, applies them via `cv2.filter2D` to the first training image, saves to `conv1_filter_effects.png`.
- **`task3.py`** — Freezes MNIST model weights, replaces `fc2` with a 3-node output layer, trains on Greek letters from `greek_train/`. Tests on personal images from `personal_greek/`.
- **`NetTransformer-template.py`** — Vision Transformer model with `PatchEmbedding`, `NetConfig`, and `NetTransformer` classes. Forward method is complete.
- **`task4.py`** — Trains the transformer on MNIST using the template. Saves model to `transformer_model.pth`.
- **`task5.py`** — Grid search over 64 configurations (4 conv1 filter counts × 4 fc1 node counts × 4 dropout rates), 3 epochs each. Saves results to `experiment_results.csv` and plots to `experiment_plots.png`.
- **`report.md`** — Full report template with all sections, tables, and image references filled in structurally. Needs actual output values and images inserted.
- **`readme.txt`** — Submission readme with file list and run instructions.

### Known issue to fix before submission
- `task1.py` plots **accuracy** per epoch, but the assignment asks for a **training/test error (loss)** plot. Consider also plotting loss, or switching the plot to loss. The example image in the spec is named `plot-trainingError.png`.

---

## What Still Needs to Be Done

### 1. Run the code and generate all outputs

Run each script in order:

```bash
# Step 1 – Train the MNIST CNN (creates mnist_model.pth, training_curves.png, test_examples.png)
python task1.py

# Step 2 – Evaluate on test set and handwritten digits (creates test_predictions.png)
# First: add your own digit images to handwritten_digits/0.png ... 9.png
python task1_test.py

# Step 3 – Examine the trained network (creates conv1_filters.png, conv1_filter_effects.png)
python task2.py

# Step 4 – Greek letter transfer learning (creates greek_training_loss.png)
# First: unzip greek_train.zip
unzip greek_train.zip
python task3.py

# Step 5 – Transformer network (creates transformer_accuracy.png, transformer_model.pth)
python task4.py

# Step 6 – Architecture experiment (creates experiment_results.csv, experiment_plots.png)
python task5.py
```

### 2. Prepare personal images

| Folder | Contents needed |
|--------|----------------|
| `handwritten_digits/` | Your own handwritten digit images named `0.png` through `9.png` (28×28 or larger, will be resized) |
| `personal_greek/alpha/` | Your own handwritten alpha images |
| `personal_greek/beta/` | Your own handwritten beta images |
| `personal_greek/gamma/` | Your own handwritten gamma images |

### 3. Fill in the report (`report.md`)

After running the code, update `report.md` with:

- [ ] Paste actual network output values into the Task 1E table
- [ ] Fill in final test accuracy for Task 1C (`~X%`)
- [ ] Describe filter observations for Task 2A (edge detectors, blobs, etc.)
- [ ] Note whether filter effects make sense for Task 2B
- [ ] Fill in how many epochs Greek transfer learning needed to reach near-perfect accuracy
- [ ] Describe personal Greek letter results
- [ ] Fill in transformer final accuracy and compare to CNN
- [ ] Fill in top/bottom configurations from Task 5 experiment
- [ ] Write discussion section for Task 5 (did results match hypotheses?)
- [ ] Fill in Extensions section (if any attempted)
- [ ] Write Reflection section (2–4 sentences)
- [ ] Replace all `[Your Name]` placeholders

### 4. Export report to PDF

The assignment requires the report submitted as a PDF:
```bash
# Using pandoc (recommended)
pandoc report.md -o report.pdf

# Or open report.md in VS Code / Typora and export to PDF
```

### 5. Submit

Per the assignment spec:
- [ ] Zip together: all `.py` files, `report.pdf`, `readme.txt`
- [ ] Upload to Canvas

---

## File Reference

```
CS5330_P5/
├── task1.py                      # Task 1A-D: CNN build + train
├── task1_test.py                 # Task 1E-F: Evaluate + handwritten
├── task2.py                      # Task 2: Filter visualization
├── task3.py                      # Task 3: Greek transfer learning
├── task4.py                      # Task 4: Transformer training
├── task5.py                      # Task 5: Architecture experiment
├── NetTransformer-template.py    # Transformer model definition
├── greek_train.zip               # Greek letter training data (unzip before task3)
├── report.md                     # Report template (fill in + export to PDF)
├── readme.txt                    # Submission readme
├── CMakeLists.txt                # C++ build config (unused for Python tasks)
└── main.cpp                      # C++ placeholder (unused for Python tasks)

Generated after running (not committed):
├── mnist_model.pth               # Trained CNN weights
├── transformer_model.pth         # Trained transformer weights
├── greek_model.pth               # Trained Greek model weights
├── training_curves.png           # Task 1C plot
├── test_examples.png             # Task 1A plot
├── test_predictions.png          # Task 1E plot
├── conv1_filters.png             # Task 2A plot
├── conv1_filter_effects.png      # Task 2B plot
├── greek_training_loss.png       # Task 3 plot
├── transformer_accuracy.png      # Task 4 plot
├── experiment_results.csv        # Task 5 raw results
└── experiment_plots.png          # Task 5 plot
```
