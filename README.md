# Classification_Support_Vector_Machines_Dataset

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Support Vector Machine Classification</div>

## 1. Project Overview
This project demonstrates a Support Vector Machine (SVM) classification using the Breast Cancer Wisconsin dataset. It includes comprehensive data preprocessing, model training, evaluation, and visualization of results.

## 1.1 Files
- `ddd.py`: Main Python script containing the SVM implementation
  - Loads the Breast Cancer Wisconsin dataset
  - Preprocesses data using standardization
  - Splits data into training and testing sets
  - Trains an SVM model with RBF kernel
  - Evaluates model performance using multiple metrics
  - Visualizes results with confusion matrix and ROC curve
- `project_answers.md`: Documentation of project findings and analysis
- `aaa.ipynb`: Jupyter notebook for interactive analysis (currently empty)
- `environment.yml`: Conda environment configuration file
- `requirements.txt`: Python package dependencies

## 1.2 Dependencies
- numpy
- matplotlib
- scikit-learn
- seaborn

## 1.3 How to Run
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python ddd.py
```

The script will:
- Train an SVM model on the Breast Cancer dataset
- Print the model's accuracy and detailed classification report
- Display a confusion matrix and ROC curve visualization
- Show feature importance analysis (for linear kernel)

## 1.4 Results
The implementation includes:
- Model evaluation metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- ROC curve with AUC score
- Feature importance analysis
