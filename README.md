# RNN Deep Learning Mini Project I
# Butterfly Image Classification using Deep Learning (RNN Mini-Project)

## Project Overview
This mini-project aims to build a robust Deep Learning model to classify different species of butterflies using image data. While image classification is traditionally dominated by Convolutional Neural Networks (CNNs), this project explores the implementation of sequential models, specifically **Recurrent Neural Networks (RNN)**, for computer vision tasks. 

By treating the rows of an image matrix as sequential timesteps (or by using a hybrid CNN-RNN architecture), the model learns to identify complex spatial patterns to distinguish between various butterfly species such as the *Monarch*, *Adonis*, and *Southern Dogface*.

## Dataset Information
The project utilizes the `butterfly-image-classification` dataset sourced from Kaggle.
*   **Total Training Data:** 6,499 images.
*   **Image Structure:** The dataset is accompanied by a `Training_set.csv` file that maps each `filename` (e.g., `Image_1.jpg`) to its actual `label`.
*   **Categories:** Contains numerous butterfly species including *GREEN CELLED CATTLEHEART*, *BROWN SIPROETA*, and many others.
*   **Train/Validation Split:** The data is split dynamically using an 80/20 ratio (80% for training, 20% for validation) to monitor generalization during training.

## Methodology & Data Preprocessing
To optimize model performance and prevent overfitting, extensive data preprocessing and augmentation were implemented using TensorFlow/Keras `ImageDataGenerator`:
1.  **Rescaling:** All pixel values were normalized by dividing by 255 (scaling them to a `0-1` range) to ensure faster and more stable model convergence.
2.  **Data Augmentation:** To make the model robust against variations in real-world images, the training generator applies real-time augmentations:
    *   `rotation_range=30`: Randomly rotates images up to 30 degrees.
    *   `width_shift` & `height_shift` (20%): Translates images vertically and horizontally.
    *   `shear_range` & `zoom_range` (20%): Applies shearing transformations and random zooming.
    *   `horizontal_flip=True`: Randomly flips images horizontally.
3.  **Target Resizing:** All images are uniformly resized to **224x224 pixels** to maintain a consistent input shape for the neural network.

---

## Detailed Analysis and Interpretation

Based on the data preparation, augmentation strategy, and the nature of the RNN architecture used in this project, here is a detailed interpretation of the workflow:

### 1. The Role of Data Augmentation in Visual Generalization
The dataset contains approximately 6,500 training images spread across multiple species. In deep learning, this amount of data can easily lead to **overfitting**—where the model memorizes the training images but fails on new, unseen data. 
By applying the `ImageDataGenerator` with aggressive augmentation (30° rotation, 20% zoom/shift/shear, and horizontal flips), we artificially expand the dataset. 
*   **Interpretation:** The model is forced to focus on the *intrinsic features* of the butterflies (like wing patterns, colors, and textures) rather than their specific position, angle, or size in the frame. This dramatically improves the model's ability to classify butterflies in the real world, where photos are often taken from imperfect angles.

### 2. Formulating Image Data for RNNs
Recurrent Neural Networks (RNN/LSTM/GRU) are inherently designed for sequence data (like time-series or text), not spatial 2D data like images. 
*   **Interpretation:** Applying an RNN to a 224x224 image implies that the image is being interpreted as a sequence. Typically, this is done by treating the image height (224) as the "timesteps" and the image width (224 x 3 color channels) as the "features" at each timestep. As the RNN processes the image row by row from top to bottom, its hidden state accumulates information about the butterfly's shape and wing patterns. This proves that sequence models can capture spatial dependencies by "scanning" an image sequentially, offering a unique alternative to standard CNN pooling methods.

### 3. Training vs. Validation Split Strategy
The `validation_split=0.2` argument reserves 20% of the training data (approx. 1,300 images) strictly for evaluating the model at the end of each epoch. 
*   **Interpretation:** Because the `valid_test_datagen` *only* applies rescaling (without the aggressive zooming, flipping, or rotating), the validation metrics reflect the model's true performance on clear, unaltered images. If the training accuracy continues to rise while the validation accuracy plateaus or drops, it serves as an early indicator of overfitting, allowing for hyperparameter tuning or early stopping.

### 4. Classification Complexity
The output from `pd.read_csv` reveals complex string labels (e.g., "GREEN CELLED CATTLEHEART"). The `flow_from_dataframe` function gracefully handles this by setting `class_mode='categorical'`, which automatically one-hot encodes these string labels into numerical arrays. 
*   **Interpretation:** The model output layer will have a number of nodes exactly matching the number of unique butterfly species found in `class_indices`. Using categorical cross-entropy loss, the network outputs a probability distribution across all possible species, making the final predictions interpretable and confident.

## How to Run the Project
1. Clone this repository: `git clone https://github.com/your-username/repo-name.git`
2. Install the required dependencies: `pip install tensorflow pandas numpy matplotlib kagglehub`
3. Open and run the Jupyter Notebook / Google Colab file `RNN-MinProject.ipynb`.
4. Ensure you have a stable internet connection for `kagglehub` to download the dataset automatically on the first run.
