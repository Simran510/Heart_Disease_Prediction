# Heart_Disease_Prediction
Heart Disease Prediction Using Machine Learning
📌 Overview
This project aims to predict the likelihood of heart disease in a patient based on various medical attributes. Using machine learning algorithms, we train a model on historical data and evaluate its performance to help in early diagnosis and preventive healthcare.

📂 Dataset
Source: UCI Heart Disease Dataset or Kaggle equivalent.
Number of Records: 303
Number of Features: 14 (including target variable)
Target Variable: target (1 = presence of heart disease, 0 = absence)

Key Features:
age - Age of the patient
sex - Gender (1 = male, 0 = female)
cp - Chest pain type
trestbps - Resting blood pressure
chol - Serum cholesterol (mg/dl)
fbs - Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)
restecg - Resting electrocardiographic results
thalach - Maximum heart rate achieved
exang - Exercise induced angina
oldpeak - ST depression induced by exercise relative to rest
slope - Slope of the peak exercise ST segment
ca - Number of major vessels (0-3) colored by fluoroscopy
thal - Thalassemia type
target - Presence (1) or absence (0) of heart disease

⚙️ Methodology
Data Preprocessing
Handle missing values
Encode categorical variables
Standardize numerical features
Exploratory Data Analysis (EDA)
Distribution plots for each feature
Correlation heatmap to find relationships
Outlier detection
Model Building

Algorithms used:
Logistic Regression
Random Forest Classifier
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Train-Test Split (e.g., 80%-20%)

Model Evaluation
Accuracy
Precision, Recall, F1-score
ROC Curve & AUC Score

🚀 Results
Best Performing Model: Random Forest Classifier (Accuracy: 0.85)

Insights:
Features like chest pain type, maximum heart rate, and ST depression have a strong influence on prediction.

🛠️ Technologies Used
Python
Pandas, NumPy (Data Manipulation)
Matplotlib, Seaborn (Data Visualization)
Scikit-learn (Model Building)

📦 How to Run the Project
Clone the repository:

bash
Copy
Edit
git clone https://github.com/username/heart-disease-prediction.git
cd heart-disease-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook or Python script:

bash
Copy
Edit
jupyter notebook heart_disease_prediction.ipynb

📌 Future Improvements
Use deep learning models for better accuracy
Deploy the model using Flask or Streamlit for real-time predictions
Include more diverse datasets for better generalization
