# Lung Cancer Detection Project

This project focuses on analyzing lung nodule malignancy and preprocessing extracted nodule images using the LIDC-IDRI dataset of CT scan DICOM images and XML radiologist annotations. The primary goal is to categorize patients into benign, malignant, or uncertain categories based on their mean malignancy scores, extract nodule images, preprocess them, and train a lung cancer detection model using a Convolutional Neural Network (CNN).

Link to dataset: [LIDC-IDRI Dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/)

---

## Notebooks Overview

### [`malignancy.ipynb`](./Notebooks/malignancy.ipynb)

The `malignancy.ipynb` notebook processes XML annotations from the LIDC-IDRI dataset to extract malignancy scores for lung nodules and categorizes patients based on their mean malignancy scores.

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

#### Outputs:
- **`malignancy_label.csv`**:
  - Contains patient IDs and their corresponding categories (benign, malignant, or uncertain).
- **`malignancy_scores.csv`**:
  - Contains detailed malignancy scores for each DICOM image.

---

### [`extract_nodule_images.ipynb`](./Notebooks/extract_nodule_images.ipynb)

The `extract_nodule_images.ipynb` notebook extracts lung nodule images from the LIDC-IDRI dataset based on the ROI (Region of Interest) coordinates provided in the XML annotations.

#### Key Features:
1. **XML Parsing**:
   - Extracts ROI coordinates from the `unblindedReadNodule` elements in the XML files.
   - Maps the ROI coordinates to DICOM images using `imageSOP_UID`.

2. **DICOM Image Matching**:
   - Searches for DICOM files corresponding to the extracted ROI coordinates.
   - Matches the `SOPInstanceUID` in the XML file with the DICOM metadata.

3. **ROI Cropping**:
   - Crops the lung nodule regions from the DICOM images based on the extracted ROI coordinates.
   - Normalizes the cropped images for consistency.

4. **Output**:
   - Saves the cropped lung nodule images to an output directory, organized by patient ID and nodule ID.

---

### [`preprocess.ipynb`](./Notebooks/preprocess.ipynb)

The `preprocess.ipynb` notebook preprocesses the extracted lung nodule images to prepare them for training a Convolutional Neural Network (CNN).

#### Key Features:
1. **Image Resizing**:
   - Resizes all images to a consistent dimension (e.g., 128x128), necessary for CNNs.

2. **Normalization**:
   - Normalizes pixel values to the range [0, 1] for faster convergence during training.

3. **Data Augmentation**:
   - Augments the dataset using techniques like rotation, flipping, and zooming to increase diversity and reduce overfitting.

4. **Class Balancing**:
   - Balances the dataset by augmenting or downsampling the minority class.

5. **Dataset Splitting**:
   - Splits the dataset into training, validation, and test sets.

6. **Saving Preprocessed Data**:
   - Saves the preprocessed datasets (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`) into a `preprocessed/` folder for future use.

---

### [`lung_cancer_detection_model.ipynb`](./Notebooks/lung_cancer_detection_model.ipynb)

The `lung_cancer_detection_model.ipynb` notebook implements the CNN model for lung cancer detection using the preprocessed datasets.

#### Key Features:
1. **Dataset Loading**:
   - Loads the preprocessed datasets (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`) from the `preprocessed/` directory.

2. **CNN Model Definition**:
   - Defines a Convolutional Neural Network (CNN) with the following architecture:
     - Two convolutional layers (`Conv2D`) with ReLU activation, followed by max-pooling layers (`MaxPooling2D`).
     - A `Flatten` layer to convert the 2D feature maps into a 1D vector.
     - A dense hidden layer with 128 neurons and ReLU activation.
     - A `Dropout` layer with a rate of 0.5 to reduce overfitting.
     - An output layer with 2 neurons (for binary classification) and a softmax activation function.

3. **Model Compilation**:
   - Optimizer: `adam` is used for efficient gradient-based optimization.
   - Loss Function: `categorical_crossentropy` is used for multi-class classification.
   - Metrics: `accuracy` and `Recall()` are used to evaluate the model's performance.

4. **Model Training**:
   - Trains the CNN model using the training and validation datasets for 20 epochs with a batch size of 32.

5. **Evaluation and Visualization**:
   - Evaluates the model on the test dataset and visualizes training/validation accuracy, loss, and recall over epochs using Matplotlib.

6. **Metrics and Confusion Matrix**:
   - Calculates precision, recall, and F1-score for each class.
   - Generates and visualizes a confusion matrix.

---

### [`pylidc.ipynb`](./Notebooks/pylidc.ipynb)

The `pylidc.ipynb` notebook explores the *pylidc* library, a Python library designed specifically for the LIDC-IDRI dataset. After experimenting with the library, it was deemed unnecessary for the project.

---

## Future Work

The preprocessed datasets and trained CNN model can be further optimized or extended for real-world applications. Additional techniques like transfer learning or hyperparameter tuning can be explored to improve model performance.