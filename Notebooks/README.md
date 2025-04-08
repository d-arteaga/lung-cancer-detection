# Lung Cancer Detection Project

This project focuses on analyzing lung nodule malignancy and ROI coordinates using the LIDC-IDRI dataset of CT scan DICOM images and XML radiologist annotations. The primary goal is to categorize patients into benign, malignant, or uncertain categories based on their mean malignancy scores derived from XML annotations, extract the nodule location from the ROI coordinates, and train a lung cancer detection model.

## Notebooks Overview

### `malignancy.ipynb`

The `malignancy.ipynb` notebook is the core of this project. It processes XML annotations from the LIDC-IDRI dataset to extract malignancy scores for lung nodules and categorizes patients based on their mean malignancy scores.

#### Key Features:
1. **XML Parsing**:
   - Extracts malignancy scores from the `unblindedReadNodule` elements in the XML files.
   - Maps malignancy scores to DICOM images using `imageSOP_UID`.

2. **Mean Malignancy Score Calculation**:
   - Calculates the mean malignancy score for each patient based on all nodules.

3. **Patient Categorization**:
   - Categorizes patients into:
     - **Benign**: Mean malignancy score between 1 and 2.
     - **Malignant**: Mean malignancy score between 4 and 5.
     - **Uncertain**: Scores outside these ranges or missing data.

4. **CSV Export**:
   - Outputs categorized patient data to `malignancy_label.csv`.
   - Saves detailed malignancy scores to `malignancy_scores.csv`.

5. **Visualization**:
   - Provides a summary of categorized patients and their malignancy scores.

#### Outputs:
- **`malignancy_label.csv`**:
  - Contains patient IDs and their corresponding categories (benign, malignant, or uncertain).
- **`malignancy_scores.csv`**:
  - Contains detailed malignancy scores for each DICOM image.

#### How to Run:
1. Ensure the LIDC-IDRI dataset is placed in the `Data/LIDC-IDRI/` directory.
2. Open the `malignancy.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Execute the cells to process the dataset and generate outputs.

#### Example Output:
```plaintext
Mean Malignancy Scores for All Patients:
  Patient 1: Mean Malignancy Score = 3.50
  Patient 2: Mean Malignancy Score = 1.75
  Patient 3: No malignancy scores available.

Patient Categories:
  Patient 1: Uncertain
  Patient 2: Benign
  Patient 3: Uncertain