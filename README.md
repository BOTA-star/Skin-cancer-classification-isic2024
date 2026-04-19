# ISIC 2024 Skin Cancer Classification

A deep learning project for skin cancer classification using the ISIC 2024 dataset.  
The model leverages both image data and metadata (multimodal learning) to improve prediction performance.

---

## Dataset

**Download dataset from:** 
https://drive.google.com/drive/folders/15o2zo7hnGKBjOfo2LfTbtulUUC5MZ-d8

### Files

- **train-metadata.csv**  

  Contains metadata and labels for the training set, used to train the model.

- **test-metadata.csv**  

  Contains metadata for the test set (no labels), used for generating predictions.

- **sample_submission.csv** 

  A template file defining the required submission format (`isic_id`, `target`) where model predictions should be filled in.

---

## Setup

Place the dataset into the following directory: 

data/

**Update paths if necessary depending on your local environment**

---

## How to Run

Follow the pipeline below:

1. **Data Preparation**
   - Load metadata (`train-metadata.csv`, `test-metadata.csv`)
   - Extract and organize image data

2. **Data Validation**
   - Check missing values, data consistency, and basic statistics

3. **Train/Validation Split**
   - Split training data into train/validation sets

4. **Feature Engineering**
   - Process image data (transforms, normalization)
   - Process metadata (encoding, scaling)

5. **Model Training**
   - Train multimodal model (image + metadata)

6. **Evaluation**
   - Evaluate performance on validation set

7. **Inference**
   - Generate predictions on test set

8. **Submission**
   - Format predictions following `sample_submission.csv`

**Warning:** in `08_test_training.ipynb`
* For quick testing:

Set `flag = True`

* For full training:

Set `flag = False`

## .py vs .ipynb
*Jupyter notebooks (.ipynb)* are used for experimentation and step-by-step execution
*Python scripts (.py)* are used for reusable modules and production-ready code
