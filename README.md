# Ethereum Fraud Detection

## Overview
This project focuses on detecting fraudulent transactions in the Ethereum network using both traditional machine learning models and deep learning techniques. By analyzing transaction attributes and interaction patterns, we aim to develop an effective fraud detection model.

## Dataset
- **Source:** Kaggle - Ethereum Fraud Detection Dataset
- **Size:** 9,842 transactions, 51 attributes
- **Target Variable:** `FLAG` (0: Valid, 1: Fraudulent)
- **Feature Categories:**
  - Transaction Timing (e.g., time difference between transactions)
  - Transaction Volume & Value (e.g., total Ether sent/received)
  - ERC20 Token Transactions
  - Interaction Patterns (e.g., unique addresses interacted with)

## Models Implemented
### Traditional Machine Learning Models:
1. **Random Forest** - Ensemble learning with decision trees
2. **Support Vector Classifier (SVC)** - Hyperplane-based classification
3. **XGBoost** - Gradient boosting for improved performance

### Deep Learning Model:
- **Fully Connected Neural Network (FCN)**:
  - Input layer with 4 neurons
  - 5 hidden layers (128 neurons each, ReLU activation)
  - Dropout (0.2) to prevent overfitting
  - Output layer with sigmoid activation
  - Trained using Adam optimizer, batch size 256, 10 epochs

## Data Processing
- Removed non-informative attributes (`Unnamed: 0`, `Index`, `Address`)
- Applied **SMOTE** for class balancing
- Feature encoding for categorical variables

## Evaluation Metrics
- **K-Fold Cross-Validation**
- **Accuracy** as primary metric
- **Confusion Matrix Analysis** for fraud detection efficiency

## Results
| Model  | Accuracy |
|--------|----------|
| Random Forest | 98.97% |
| SVC | 95.00% |
| XGBoost | 99.00% |
| FCN (Tuned) | 98.72% |

## Key Findings
- **XGBoost outperformed traditional models**, achieving the highest accuracy.
- **FCN performed competitively**, demonstrating deep learning’s potential for fraud detection.
- **SMOTE successfully balanced the dataset**, improving fraud classification.
- **Hyperparameter tuning improved FCN performance**, optimizing layers, dropout rate, and batch size.

## Future Work
- **Feature Engineering** to enhance transaction pattern detection
- **Exploring Graph Neural Networks (GNNs)** for analyzing Ethereum transaction networks
- **Recurrent Neural Networks (RNNs)** for sequential transaction modeling

## Installation & Usage
### Requirements:
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`

### Running the Project:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ethereum-fraud-detection.git
   cd ethereum-fraud-detection
