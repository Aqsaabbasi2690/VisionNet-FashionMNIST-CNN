Image Classification using Upgraded Convolutional Neural Network (CNN)
# Project Overview

This project implements an Upgraded Convolutional Neural Network (CNN) for image classification.
The model is designed to improve performance over a basic CNN by incorporating:

Batch Normalization

Dropout Regularization

Data Augmentation

Improved Architecture Depth

Adam Optimizer

Early Stopping

The goal is to achieve better generalization and higher validation accuracy while preventing overfitting.

Model Architecture

The CNN consists of:

Multiple Conv2D layers

BatchNormalization after convolution

MaxPooling layers

Dropout layers

Fully Connected Dense layers

Softmax output layer

Architecture Flow:

Input Image
→ Conv2D + ReLU
→ BatchNorm
→ MaxPooling
→ Dropout
→ Conv2D + ReLU
→ BatchNorm
→ MaxPooling
→ Dropout
→ Flatten
→ Dense
→ Dropout
→ Output Layer (Softmax)

📂 Project Structure

VisionNet-FashionMNIST-CNN

├── fashion_cnn.ipynb

├── best_fashion_cnn.keras

├── fashion_classifier_production.keras

├── requirements.txt

├── README.md


📊 Dataset

Images are organized in class-wise folders.

Supports multi-class classification.

Data augmentation is applied during training to improve robustness.

Example structure:

dataset/
   ├── class_1/
   ├── class_2/
   ├── class_3/

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install dependencies:

pip install -r requirements.txt


Model will:

Train on training data

Validate on validation data

Save the best model

📈 Results

Improved validation accuracy compared to basic CNN

Reduced overfitting using Dropout

Stable training using Batch Normalization

Example:

Training Accuracy: 94%
Validation Accuracy: 91%

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

🔍 Key Improvements Over Basic CNN
| Feature             | Basic CNN | Upgraded CNN |
| ------------------- | --------- | ------------ |
| Batch Normalization | ❌         | ✅            |
| Dropout             | ❌         | ✅            |
| Data Augmentation   | ❌         | ✅            |
| Regularization      | Limited   | Improved     |
| Overfitting Control | Weak      | Strong       |

🎯 Future Improvements

Transfer Learning (MobileNet / EfficientNet)

Hyperparameter Tuning

Model Deployment (Streamlit / Flask)

Confusion Matrix & Classification Report

👩‍💻 Author

Aqsa Abbasi

⭐ If You Like This Project

Give this repository a ⭐ and connect with me on LinkedIn!
