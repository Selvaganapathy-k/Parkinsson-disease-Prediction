Subject: README.md – Parkinson’s Disease Prediction (Updated App URL)

# 🧠 Parkinson’s Disease Prediction

## 📌 Project Description

This project implements a **Parkinson’s Disease Prediction system** using Machine Learning.
The model predicts whether a person is **affected by Parkinson’s disease** or **healthy** based on biomedical voice measurements.

The project includes:

* Data analysis and preprocessing
* Model training and selection
* A **Streamlit web application** for real-time disease prediction

This project is developed as a **mini project** for academic learning and practical exposure to healthcare-related machine learning applications.

---

## 📁 Dataset Information

* **Dataset Name:** Parkinson’s Disease Dataset
* **File:** `Parkinsson disease.csv`

The dataset contains biomedical voice measurements such as:

* Fundamental frequency measures
* Jitter and shimmer values
* Noise-to-harmonics ratio
* Nonlinear dynamical complexity measures
* Health status (target variable)

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit

---

## 📂 Project Structure

```
Parkinsson-disease-Prediction
│
├── Parkinsson disease.csv
├── parkinsson_disease_pre.ipynb
├── best_model.pkl
├── scaler.pkl
├── columns.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Selvaganapathy-k/Parkinsson-disease-Prediction
cd Parkinsson-disease-Prediction
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application Locally

```bash
streamlit run app.py
```

---

## 🌐 Live Application

🔗 **Streamlit App URL:**
👉 [https://parkinappn-disease-prediction-4fdtb5m6hucjk5tipmxeat.streamlit.app/](https://parkinappn-disease-prediction-4fdtb5m6hucjk5tipmxeat.streamlit.app/)

---

## 🔍 Model Details

* Problem Type: **Binary Classification**
* Target Variable:

  * `1` → Parkinson’s Disease
  * `0` → Healthy
* Data Preprocessing:

  * Feature scaling using **StandardScaler**
  * Handling class imbalance
* Model Selection:

  * Best-performing model saved as `best_model.pkl`

---

## 📈 Features

* User-friendly Streamlit interface
* Accepts biomedical voice feature inputs
* Real-time prediction with probability score
* Uses saved scaler and feature columns
* Reproducible and reliable predictions

---

## 🎓 Learning Outcomes

* Understanding healthcare ML problems
* Handling imbalanced datasets
* Feature scaling and preprocessing
* Training and evaluating classification models
* Saving and loading ML models
* Deploying ML applications using Streamlit

---

## 📌 Notes

* Virtual environment folders (`venv`, `myvenv`) are not included in the repository.
* All dependencies are listed in `requirements.txt`.
* ⚠️ This project is for **educational purposes only** and not for medical diagnosis.

---

## ✍️ Author

**Selvaganapathy K**
Computer Science Student

---

## 🏁 Conclusion

This project demonstrates how machine learning techniques can assist in early detection of Parkinson’s disease using biomedical data and interactive web applications.

