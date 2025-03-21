# TensorFlow MNIST Neural Network Project

## Overview
This project demonstrates a simple yet effective neural network using TensorFlow to classify handwritten digits from the MNIST dataset. It covers data preprocessing, model building, training with callbacks, evaluation, and detailed visualizations of the results. The generated plots (training history, confusion matrix, and sample predictions) are automatically saved to the `plots` folder.

## Features
- **Data Preprocessing:** Loads the MNIST dataset and normalizes the images.
- **Model Architecture:** A fully connected neural network with dropout for regularization.
- **Callbacks:** Implements early stopping and model checkpointing to optimize training.
- **Visualization:** 
  - Training and validation accuracy/loss plots.
  - A normalized confusion matrix.
  - A grid of sample predictions with true vs. predicted labels.
- **Modular Code:** Organized functions for ease of maintenance and future improvements.

## Requirements
- **Python:** Version 3.7 to 3.10 (TensorFlow is not yet supported on Python 3.13)
- **Libraries:**
  - TensorFlow
  - Matplotlib
  - NumPy
  - Scikit-learn

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd Tensorflow
   ```
2. **Create a Virtual Environment (Recommended):**

   ```bash
    python -m venv tf-env
   ```
Activate the environment:

- Windows:
   ```bash
    tf-env\Scripts\activate
   ```
- macOS/Linux:
   ```bash
    source tf-env/bin/activate
   ```
3. **Install Dependencies:**

   ```bash
    pip install tensorflow matplotlib numpy scikit-learn
   ```
## Running the Project
Run the main script to train and evaluate the model and generate the plots:

   ```bash
    python <your_project_filename>.py
   ```
After running, check the plots folder for the following saved images:

- training_history.png: Shows training/validation accuracy and loss over epochs.
- confusion_matrix.png: A normalized confusion matrix of the test predictions.
- sample_predictions.png: A grid of randomly selected test images with their true and predicted labels.

## Project Structure

   ```bash
    Tensorflow/
    ├── main.py         # Main Python script containing the project code
    ├── plots/          # Directory where generated plots are saved
    ├── README.md       # This file
    └── LICENSE         # License file
   ```
## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## Acknowledgments
- TensorFlow: For providing an excellent deep learning framework.
- MNIST Dataset: For being a benchmark dataset in image classification.
- Open-Source Community: For continuous support and contributions.

## License
This project is licensed under the MIT License – see the LICENSE file for details.