# ISIC 2024 Skin Cancer Classification

This project is a multimodal deep learning pipeline for skin cancer classification using the ISIC 2024 dataset.

The model uses both:

* Skin lesion images
* Patient and lesion metadata

to predict the probability that a skin lesion is malignant.

This repository is currently configured for a lightweight local CPU demo. Full training can be performed later on Kaggle or Google Colab with GPU.

> This project is developed for academic and research purposes only. It is not intended for real-world medical diagnosis.

---

## 1. Project Overview

Skin cancer classification is challenging because malignant samples are rare and visual differences between benign and malignant lesions can be subtle.

Instead of using only images, this project applies a multimodal learning approach:

```text
Image input    → CNN backbone
Metadata input → MLP branch
Image features + Metadata features → Fusion classifier → Prediction
```

Current demo architecture:

```text
MobileNetV3-Small + Metadata MLP
```

The current local version is designed to:

* run on a personal computer without GPU
* validate the full machine learning pipeline
* demonstrate the project structure clearly on GitHub
* prepare the codebase for future full training on GPU

---

## 2. Dataset

Dataset used: **ISIC 2024 Skin Cancer Detection Challenge**

Download dataset from:

```text
https://drive.google.com/drive/folders/15o2zo7hnGKBjOfo2LfTbtulUUC5MZ-d8
```

Expected files:

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

Expected local structure:

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

Large data files are not included in this repository.

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
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Generated files are ignored by Git:

```text
data/
checkpoints/
artifacts/
outputs/
models/
```

---

## 4. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

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

The current configuration is optimized for local CPU demo mode:

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

This setup is intentionally lightweight. It is used to verify that the whole pipeline works correctly on a personal computer.

For full training on Kaggle or Colab GPU, these values can be increased later.

---

## 6. Run the Full Pipeline

The easiest way to run the project is:

```bash
python main.py
```

This runs the full pipeline:

```text
01_data_integrity.py
02_split_data.py
03_preprocess_metadata.py
04_test_dataloader.py
05_test_model.py
06_train_demo.py
07_evaluate_holdout.py
08_infer_submission.py
```

The `main.py` file acts as a pipeline runner. It executes each script step by step and stops if any step fails.

---

## 7. Run Individual Stages

Run only dataset checking:

```bash
python main.py --stage check
```

Run data splitting and metadata preprocessing:

```bash
python main.py --stage prepare
```

Run dataloader and model verification:

```bash
python main.py --stage verify
```

Run demo training:

```bash
python main.py --stage train
```

Run holdout evaluation:

```bash
python main.py --stage evaluate
```

Run inference and generate submission:

```bash
python main.py --stage infer
```

Run local demo without inference:

```bash
python main.py --stage demo
```

Preview the scripts without executing them:

```bash
python main.py --stage all --dry-run
```

---

## 8. Pipeline Details

### Step 1: Data Integrity Check

```bash
python scripts/01_data_integrity.py
```

Checks:

* train metadata
* test metadata
* required columns
* image folder
* image path consistency
* target distribution

---

### Step 2: Data Split

```bash
python scripts/02_split_data.py
```

Creates:

```text
data/splits/train_split.csv
data/splits/val_split.csv
data/splits/holdout_split.csv
```

The split is performed by `patient_id` to reduce patient-level data leakage.

---

### Step 3: Metadata Preprocessing

```bash
python scripts/03_preprocess_metadata.py
```

Creates:

```text
data/processed/train_processed.csv
data/processed/val_processed.csv
data/processed/holdout_processed.csv
artifacts/metadata_preprocessor.pkl
artifacts/metadata_features.json
```

Metadata preprocessing includes:

* numeric imputation
* standard scaling
* categorical imputation
* one-hot encoding

The preprocessor is fitted only on the training set.

---

### Step 4: Dataloader Test

```bash
python scripts/04_test_dataloader.py
```

Expected tensor shapes:

```text
Image batch: [batch_size, 3, image_size, image_size]
Meta batch : [batch_size, metadata_dim]
Label batch: [batch_size]
```

---

### Step 5: Model Test

```bash
python scripts/05_test_model.py
```

Checks whether the multimodal model can process:

```text
image tensor + metadata tensor
```

and output:

```text
logits with shape [batch_size, 1]
```

---

### Step 6: Demo Training

```bash
python scripts/06_train_demo.py
```

Creates:

```text
checkpoints/best.pth
checkpoints/last.pth
outputs/train_demo_history.csv
outputs/train_demo_metrics.json
```

---

### Step 7: Holdout Evaluation

```bash
python scripts/07_evaluate_holdout.py
```

Creates:

```text
outputs/holdout_metrics.json
```

---

### Step 8: Inference and Submission

```bash
python scripts/08_infer_submission.py
```

Creates:

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

## 9. Model Architecture

The demo model uses two input branches.

### Image Branch

```text
Input image
→ MobileNetV3-Small
→ Image feature vector
```

### Metadata Branch

```text
Input metadata
→ Fully connected layers
→ Metadata feature vector
```

### Fusion Classifier

```text
Image feature vector + Metadata feature vector
→ Concatenate
→ Dense layers
→ Malignancy probability
```

Overall flow:

```text
Image
  ↓
CNN Backbone
  ↓
Image Features
        \
         → Concatenate → Classifier → Prediction
        /
Metadata
  ↓
MLP
  ↓
Metadata Features
```

---

## 10. Metadata Features

The model uses metadata columns that are available in both train and test data, such as:

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
```

Columns that may leak diagnosis information or are not available at prediction time are not used as input features, for example:

```text
target
patient_id
lesion_id
iddx_full
iddx_1
iddx_2
iddx_3
iddx_4
iddx_5
mel_mitotic_index
mel_thick_mm
```

`patient_id` is used only for data splitting, not as a model input.

---

## 11. Demo Results

The current CPU demo uses:

```text
Backbone: MobileNetV3-Small
Image size: 160
Batch size: 4
Epochs: 2
Max train batches: 20
Max validation batches: 10
Use pretrained weights: False
```

Example demo results:

```text
Validation AUC: 0.6520
Validation PR-AUC: 0.5970

Holdout AUC: 0.6983
Holdout PR-AUC: 0.3722
```

These results are from a lightweight CPU demo and should not be considered final model performance.

The goal of this version is to prove that the full pipeline works end-to-end:

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

## 12. Local Demo vs Full Training

### Local CPU Demo

Used for:

```text
testing
debugging
GitHub demonstration
academic report validation
```

Characteristics:

```text
small subset of data
small image size
MobileNetV3-Small
no pretrained weights
limited training batches
```

### Full GPU Training

For future training on Kaggle or Colab:

```python
IMAGE_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 15

NEGATIVE_SAMPLE_SIZE = 50000
EVAL_NEGATIVE_SAMPLE_SIZE = 10000

BACKBONE = "efficientnet_b3"
USE_PRETRAINED = True
```

Possible improvements for the full version:

* use pretrained EfficientNet-B0 or EfficientNet-B3
* increase image size
* train for more epochs
* use stronger augmentation
* tune class imbalance handling
* apply focal loss
* tune prediction threshold
* add Grad-CAM explainability
* compare image-only, metadata-only, and multimodal models

---

## 13. Important Notes

The ISIC 2024 dataset is highly imbalanced. In the raw training metadata, malignant samples are much fewer than benign samples.

Because of this, accuracy alone is not a reliable metric. This project reports:

```text
AUC
PR-AUC
Precision
Recall
F1-score
```

The local demo model may predict poorly at a fixed threshold of `0.5`, but AUC can still show whether the model is learning useful ranking behavior.

---

## 14. GitHub Notes

Do not commit large data, model checkpoints, or generated outputs.

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
main.py
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

## 15. Future Work

Future improvements include:

* full training on Kaggle or Google Colab GPU
* using pretrained CNN backbones
* comparing multiple backbones
* adding image-only and metadata-only baselines
* improving class imbalance strategy
* adding threshold optimization
* adding Grad-CAM visualization
* adding experiment tracking
* improving model interpretation for academic reporting

---

## 16. Disclaimer

This project is for academic and educational purposes only.

The model predictions should not be used for medical diagnosis or clinical decision-making.
