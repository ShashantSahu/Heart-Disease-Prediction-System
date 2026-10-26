# ❤️ CardioPro - Heart Disease Prediction System

### Machine Learning-Based Heart Disease Risk Prediction

A web-based machine learning project designed to analyze health-related input features and predict the possibility of heart disease using trained classification models. Built with **Python, Streamlit, Pandas, NumPy, and Scikit-learn**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-2E8B57?style=for-the-badge" alt="Machine Learning">
</p>

🌐 **Live Website:** [CardioPro - Heart Disease Prediction System
](https://heart-disease-prediction-system-vsuasxtlermupywpb2xe8v.streamlit.app/)


## 📌 Overview

The Heart Disease Prediction System demonstrates how machine learning can be applied to health-related data classification.

The application uses trained machine learning models to process user-provided input features and generate a prediction. Its web interface is designed to make the prediction workflow accessible and easy to understand.

### 🎯 Project Objectives

- Apply machine learning techniques to heart disease prediction.
- Process health-related features for classification.
- Provide an interactive web-based interface.
- Explore and compare trained classification models.
- Demonstrate a practical application of Python and machine learning.

## ✨ Key Features

- ❤️ Heart disease prediction using trained ML models
- 🤖 Multiple classification models
- 🖥️ Interactive Streamlit web application
- 📊 Health-related input feature processing
- 📈 Model evaluation and performance analysis
- 🗂️ Trained model files for application inference
- 📱 User-friendly prediction interface

*The exact features and model evaluation options depend on the current implementation in the repository.*

## 🤖 Machine Learning Models

The project includes the following trained models:

- Decision Tree
- Extra Trees
- Gradient Boosting
- Logistic Regression
- Random Forest

These models represent different classification approaches. Their performance should be compared using the project's actual evaluation results rather than assuming one model is always best.

## 🔄 Machine Learning Workflow

```text
Health-Related Input Features
             ↓
     Data Preprocessing
             ↓
     Feature Preparation
             ↓
   Trained ML Classifier
             ↓
    Prediction Generation
             ↓
      Streamlit Interface
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and classification |
| Joblib | Saving and loading trained models |

## 📂 Project Structure

```text
Heart-Disease-Prediction-System/
│
├── app.py
├── models/
│   ├── decision_tree.joblib
│   ├── extra_trees.joblib
│   ├── gradient_boosting.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── metadata.json
│
├── requirements.txt
├── README.md
└── assets/
    └── screenshots/
```

*This is a representative structure. Adjust filenames and folders to match the actual repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- pip package manager
- Git (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/ShashantSahu/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local application in your browser at:

`http://localhost:8501`

## 🖼️ Application Screenshots

Add screenshots of the actual running application to showcase its interface.

Suggested screenshots:

- Main dashboard
- Prediction input form
- Prediction results
- Model performance dashboard

Once the screenshot files are uploaded, display them using Markdown:

```markdown
![Application Dashboard](assets/screenshots/dashboard.png)

![Prediction Results](assets/screenshots/prediction-results.png)
```

## 📊 Model Evaluation

Model performance should be assessed using appropriate evaluation metrics on held-out test data.

Potential metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases identified.
- **F1-score:** Balance between precision and recall.
- **Confusion Matrix:** Breakdown of correct and incorrect classifications.

Add actual values from the project's evaluation results before publishing performance claims.

## 🚀 Deployment

The application is designed to run using Streamlit and can be deployed through Streamlit Community Cloud.

**Repository:** [Heart Disease Prediction System on GitHub](https://github.com/ShashantSahu/Heart-Disease-Prediction-System)

Add the live application URL here once the deployment is confirmed working.

## 🔮 Future Scope

- Improve model evaluation and validation.
- Add explainable AI to help interpret model predictions.
- Enhance dashboard visualizations.
- Maintain prediction history where appropriate and secure.
- Generate downloadable prediction reports.
- Improve input validation and application usability.

These are potential enhancements and are not necessarily implemented in the current version.

## ⚠️ Medical Disclaimer

This project is intended for **educational and demonstration purposes only**. Its predictions are not a medical diagnosis and must not replace professional medical advice, clinical evaluation, or emergency care.

Model predictions can be incorrect and may be affected by the quality and limitations of the training data. Consult a qualified healthcare professional for medical concerns.

## 👨‍💻 Developer

**Shashant Sahu**

B.Tech — Computer Science and Engineering

Interests: Machine Learning · Python · Data Science · AI Applications

---

<p align="center">
  <b>Built with Python, Machine Learning, and Streamlit.</b>
  <br>
  If you find this project interesting, consider giving the repository a ⭐.
</p>
