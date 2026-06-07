# ISIC 2024 Skin Cancer Classification

This project implements a multimodal deep learning pipeline for skin cancer classification using the ISIC 2024 dataset.

The model combines:

* Skin lesion images
* Patient and lesion metadata

to predict the probability of a lesion being malignant.

> Note: This project is for academic and research purposes only. It is not intended for real-world medical diagnosis.

---

## 1. Project Overview

Skin cancer classification is a challenging computer vision problem because the dataset is highly imbalanced and visual differences between benign and malignant lesions can be subtle.

To improve the prediction pipeline, this project uses a multimodal learning approach:

```text
Image input      → CNN backbone
Metadata input   → MLP branch
Image features + Metadata features → Fusion classifier → Prediction
```

Current local demo model:

```text
MobileNetV3-Small + Metadata MLP
```

The local version is designed to run on CPU with a small subset of the dataset.
Full training can be performed later on Kaggle or Colab with GPU.

---

## 2. Dataset

Dataset used: ISIC 2024 Skin Cancer Detection Challenge

Download dataset from:

```text
https://drive.google.com/drive/folders/15o2zo7hnGKBjOfo2LfTbtulUUC5MZ-d8
```

Expected dataset files:

```text
train-metadata.csv
test-metadata.csv
sample_submission.csv
ISIC_2024_Training_Input.zip
test-image.hdf5
```

After downloading, place the dataset under:

```text
data/raw/
```

Expected structure:

```text
DATN/
├── data/
│   └── raw/
│       ├── train-metadata.csv
│       ├── test-metadata.csv
│       ├── sample_submission.csv
│       ├── ISIC_2024_Training_Input.zip
│       ├── test-image.hdf5
│       └── ISIC_2024_Training_Input/
│           ├── ISIC_0015670.jpg
│           ├── ISIC_0015845.jpg
│           └── ...
```

The image folder can be extracted automatically or manually from:

```text
ISIC_2024_Training_Input.zip
```

---

## 3. Project Structure

```text
DATN/
├── config/
│   └── config.py
│
├── src/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── metadata.py
│   ├── model.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
│
├── scripts/
│   ├── 01_data_integrity.py
│   ├── 02_split_data.py
│   ├── 03_preprocess_metadata.py
│   ├── 04_test_dataloader.py
│   ├── 05_test_model.py
│   ├── 06_train_demo.py
│   ├── 07_evaluate_holdout.py
│   └── 08_infer_submission.py
│
├── notebooks/
│   └── legacy notebooks / experiments
│
├── requirements.txt
├── README.md
└── .gitignore
```

Generated files are not committed to GitHub:

```text
data/
checkpoints/
artifacts/
outputs/
models/
```

---

## 4. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
torch
torchvision
pandas
numpy
scikit-learn
Pillow
h5py
tqdm
joblib
matplotlib
```

---

## 5. Configuration

Main configuration file:

```text
config/config.py
```

For local CPU demo mode, the project uses a lightweight setup:

```python
IMAGE_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 2

NEGATIVE_SAMPLE_SIZE = 500
EVAL_NEGATIVE_SAMPLE_SIZE = 200

MAX_TRAIN_BATCHES = 20
MAX_VAL_BATCHES = 10

BACKBONE = "mobilenet_v3_small"
USE_PRETRAINED = False
NUM_WORKERS = 0
```

This mode is suitable for:

* testing the full pipeline
* running on a personal computer without GPU
* demonstrating the project on GitHub

For full training on Kaggle or Colab GPU, these values can be increased later.

---

## 6. How to Run

Run the pipeline step by step.

### Step 1: Check dataset integrity

```bash
python scripts/01_data_integrity.py
```

This checks:

* metadata files
* required columns
* image folder
* image path consistency
* train/test metadata structure

---

### Step 2: Split data

```bash
python scripts/02_split_data.py
```

This creates:

```text
data/splits/train_split.csv
data/splits/val_split.csv
data/splits/holdout_split.csv
```

The split is performed by `patient_id` to avoid patient-level data leakage.

---

### Step 3: Preprocess metadata

```bash
python scripts/03_preprocess_metadata.py
```

This creates:

```text
data/processed/train_processed.csv
data/processed/val_processed.csv
data/processed/holdout_processed.csv
artifacts/metadata_preprocessor.pkl
artifacts/metadata_features.json
```

Metadata preprocessing includes:

* numeric imputation
* numeric scaling
* categorical imputation
* one-hot encoding

The preprocessor is fitted only on the training set.

---

### Step 4: Test dataloader

```bash
python scripts/04_test_dataloader.py
```

Expected output:

```text
Image batch shape: [batch_size, 3, image_size, image_size]
Meta batch shape : [batch_size, metadata_dim]
Label batch shape: [batch_size]
```

---

### Step 5: Test model forward pass

```bash
python scripts/05_test_model.py
```

This verifies that the multimodal model can receive:

```text
image tensor + metadata tensor
```

and produce:

```text
logits with shape [batch_size, 1]
```

---

### Step 6: Train demo model

```bash
python scripts/06_train_demo.py
```

This trains a lightweight demo model using a small subset of the dataset.

Generated files:

```text
checkpoints/best.pth
checkpoints/last.pth
outputs/train_demo_history.csv
outputs/train_demo_metrics.json
```

Example local demo result:

```text
Best validation AUC: 0.6520
```

---

### Step 7: Evaluate on holdout set

```bash
python scripts/07_evaluate_holdout.py
```

Generated file:

```text
outputs/holdout_metrics.json
```

Example local demo result:

```text
Holdout AUC: 0.6983
PR-AUC: 0.3722
```

---

### Step 8: Generate submission file

```bash
python scripts/08_infer_submission.py
```

Generated file:

```text
outputs/submission.csv
```

Submission format:

```text
isic_id,target
ISIC_0015657,0.519349
ISIC_0015729,0.520886
ISIC_0015740,0.518118
```

---

## 7. Model Architecture

The current demo model uses:

```text
Image branch:
MobileNetV3-Small

Metadata branch:
Fully connected neural network

Fusion:
Concatenate image features and metadata features

Classifier:
Dense layers → binary prediction
```

Architecture summary:

```text
Image
  ↓
CNN Backbone
  ↓
Image Feature Vector
        \
         → Concatenate → Classifier → Malignancy Probability
        /
Metadata
  ↓
MLP
  ↓
Metadata Feature Vector
```

---

## 8. Metadata Features

The project uses metadata columns that are available in both train and test data, including:

```text
age_approx
sex
anatom_site_general
clin_size_long_diam_mm
tbp_lv_A
tbp_lv_B
tbp_lv_C
tbp_lv_H
tbp_lv_L
tbp_lv_areaMM2
tbp_lv_perimeterMM
tbp_lv_location
tbp_lv_location_simple
...
```

Columns related to diagnosis or unavailable test labels are not used as input features, such as:

```text
target
iddx_full
iddx_1
iddx_2
iddx_3
iddx_4
iddx_5
mel_mitotic_index
mel_thick_mm
lesion_id
patient_id
```

`patient_id` is used only for splitting data, not as a model input.

---

## 9. Local Demo vs Full Training

### Local demo mode

Used for:

```text
CPU testing
GitHub demonstration
fast end-to-end pipeline validation
```

Characteristics:

```text
Small data subset
Small image size
MobileNetV3-Small
No pretrained weights
Limited training batches
```

### Full training mode

For future GPU training on Kaggle or Colab:

```text
Use larger dataset sample or full dataset
Use pretrained CNN backbone
Increase image size
Increase batch size if GPU memory allows
Train for more epochs
Consider EfficientNet-B0 / EfficientNet-B3
Tune imbalance handling
```

Example full-training direction:

```python
IMAGE_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 15
NEGATIVE_SAMPLE_SIZE = 50000
EVAL_NEGATIVE_SAMPLE_SIZE = 10000
BACKBONE = "efficientnet_b3"
USE_PRETRAINED = True
```

---

## 10. Results

Current local CPU demo result:

```text
Validation AUC: 0.6520
Holdout AUC: 0.6983
Holdout PR-AUC: 0.3722
```

These results are from a lightweight CPU demo configuration and are not final model performance.

The purpose of this version is to verify that the full pipeline works correctly:

```text
data validation
→ patient-level split
→ metadata preprocessing
→ multimodal dataloader
→ multimodal model
→ training
→ evaluation
→ inference
→ submission
```

---

## 11. Notes

* The dataset is highly imbalanced.
* Accuracy alone is not a reliable metric.
* AUC and PR-AUC are more informative for this task.
* The local demo model may predict poorly at threshold `0.5`, but AUC can still show whether the model is learning ranking ability.
* Full performance should be evaluated after GPU training.

---

## 12. GitHub Notes

Do not commit large data or model files.

Recommended `.gitignore`:

```gitignore
data/
checkpoints/
artifacts/
outputs/
models/

*.pth
*.pt
*.ckpt
*.zip
*.hdf5

__pycache__/
*.pyc
.ipynb_checkpoints/

venv/
.env
```

Recommended files to commit:

```text
config/
src/
scripts/
notebooks/
requirements.txt
README.md
.gitignore
```

Do not commit:

```text
data/
checkpoints/
artifacts/
outputs/
venv/
```

---

## 13. Future Improvements

Potential improvements:

* Train full model on Kaggle or Colab GPU
* Use pretrained EfficientNet-B0 or EfficientNet-B3
* Add focal loss for class imbalance
* Add test-time augmentation
* Tune decision threshold
* Add Grad-CAM visualization for explainability
* Compare image-only, metadata-only, and multimodal models
* Add experiment tracking for multiple configurations

---

## 14. Disclaimer

This project is developed for academic research and educational purposes only.
The model output should not be used as a medical diagnosis or clinical decision-making tool.
