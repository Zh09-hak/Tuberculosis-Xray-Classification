# 🩺 Tuberculosis Detection from Chest X-rays

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview

This project uses deep learning to detect tuberculosis from chest X-ray images.

---

## 📊 Dataset

Dataset from Kaggle:

https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

⚠️ Dataset is not included due to size limitations.

---

## 🧠 Model Architecture

* Convolutional Neural Network (CNN)
* Multiple Conv2D + MaxPooling layers
* Fully connected layers with Dropout

---

## 📈 Results

* Training Accuracy: 99%
* Validation Accuracy: 97.8%

---

## ⚙️ Project Structure

```id="j4l87q"
project/
├── src/
├── data/ (not included)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```id="0nq9ga"
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

---

## 🔮 Example

* TB image → 0.99
* Normal image → 0.05

---

## ⚠️ Disclaimer

This model is for educational purposes only and should not be used for medical diagnosis.

---

## 👨‍💻 Author

https://github.com/Zh09-hak
