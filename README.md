# Heart Disease Prediction Project

## Overview

This project aims to predict the presence of heart disease in patients using machine learning algorithms. The model is built using a dataset containing various health metrics and indicators, which are processed and analyzed to determine the likelihood of heart disease. The goal is to provide a reliable tool that can assist healthcare professionals in diagnosing heart disease early and accurately.

## Features

- **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizing and understanding the data distribution and relationships between features.
- **Feature Engineering**: Selecting and creating relevant features to improve model performance.
- **Model Training**: Implementing and training multiple machine learning algorithms.
- **Model Evaluation**: Assessing model performance using various metrics and selecting the best-performing model.
- **Prediction**: Using the trained model to predict heart disease on new data.

## Dataset

The dataset used in this project is the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), which contains 303 instances and 14 attributes, including age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, rest ECG results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage

Open the `heart_disease_prediction.ipynb` notebook in Jupyter and follow the steps to preprocess the data, perform EDA, train the models, and make predictions. Each section of the notebook contains detailed instructions and explanations.

## Results

The project evaluates several machine learning models, including Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines. The models are compared based on accuracy, precision, recall, and F1-score. The best-performing model is saved and can be used to make predictions on new data.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Acknowledgements

This project is based on the Heart Disease UCI dataset available at the UCI Machine Learning Repository. Special thanks to the contributors of this dataset and the open-source community for providing tools and resources that made this project possible.

