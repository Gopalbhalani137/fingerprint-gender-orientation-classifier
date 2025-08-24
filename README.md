# Fingerprint Gender & Orientation Classifier

This project classifies **gender, hand orientation, and finger type** from fingerprint images.  
We use a **CNN for feature extraction** and a **Random Forest** classifier for final predictions.  

Dataset: **[SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing?utm_source=chatgpt.com)**

---

## 🚀 Features
- CNN extracts robust fingerprint features
- Random Forest performs:
  - Gender classification (Male / Female)
  - Hand classification (Left / Right)
  - Finger type classification (Thumb, Index, Middle, Ring, Little)
- Achieves **96–99% accuracy**

---

## 📊 Results

### Gender Classification
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Male    | 0.97      | 0.99   | 0.98     |
| Female  | 0.94      | 0.85   | 0.90     |
| **Accuracy** | **0.96** |

### Hand Classification
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Left    | 0.99      | 0.98   | 0.99     |
| Right   | 0.98      | 0.99   | 0.99     |
| **Accuracy** | **0.99** |

### Finger Classification
| Finger  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Thumb   | 1.00      | 0.98   | 0.99     |
| Index   | 0.98      | 0.97   | 0.97     |
| Middle  | 0.95      | 0.94   | 0.94     |
| Ring    | 0.96      | 0.95   | 0.95     |
| Little  | 0.95      | 0.98   | 0.96     |
| **Accuracy** | **0.96** |

---

## 📚 Comparison with Research

| Study / Model | Dataset | Accuracy |
|---------------|---------|----------|
| Narayanan & Sajith, 2019 – Pixel Count | SOCOFing | Male: 96.4%, Female: 90.2% ([SSRN](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3471862_code3562725.pdf?abstractid=3444032&mirid=1&utm_source=chatgpt.com)) |
| Giudice et al., 2020 – Inception-v3 multi-task | SOCOFing | Gender: 92.5%, Hand: 97.5%, Finger: 92.1% ([arXiv](https://arxiv.org/abs/2007.04931?utm_source=chatgpt.com)) |
| Qi et al., 2021 – DDC-ResNet Autoencoder | SOCOFing | Avg: 96.5%, Male: 97.5%, Female: 95.5% ([arXiv](https://arxiv.org/abs/2108.08233?utm_source=chatgpt.com)) |
| Alhijaj & Khudeyer, 2023 – EfficientNetB0 + RF | SOCOFing | 99.91% ([IIETA](https://iieta.org/download/file/fid/153118?utm_source=chatgpt.com)) |
| **Our Model – CNN + Random Forest** | SOCOFing | **96–99%** |

---

## 🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, seaborn

---

## 📂 Project Structure
# Fingerprint Gender & Orientation Classifier

This project classifies **gender, hand orientation, and finger type** from fingerprint images.  
We use a **CNN for feature extraction** and a **Random Forest** classifier for final predictions.  

Dataset: **[SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing?utm_source=chatgpt.com)**

---

## 🚀 Features
- CNN extracts robust fingerprint features
- Random Forest performs:
  - Gender classification (Male / Female)
  - Hand classification (Left / Right)
  - Finger type classification (Thumb, Index, Middle, Ring, Little)
- Achieves **96–99% accuracy**

---

## 📊 Results

### Gender Classification
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Male    | 0.97      | 0.99   | 0.98     |
| Female  | 0.94      | 0.85   | 0.90     |
| **Accuracy** | **0.96** |

### Hand Classification
| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Left    | 0.99      | 0.98   | 0.99     |
| Right   | 0.98      | 0.99   | 0.99     |
| **Accuracy** | **0.99** |

### Finger Classification
| Finger  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Thumb   | 1.00      | 0.98   | 0.99     |
| Index   | 0.98      | 0.97   | 0.97     |
| Middle  | 0.95      | 0.94   | 0.94     |
| Ring    | 0.96      | 0.95   | 0.95     |
| Little  | 0.95      | 0.98   | 0.96     |
| **Accuracy** | **0.96** |

---

## 📚 Comparison with Research

| Study / Model | Dataset | Accuracy |
|---------------|---------|----------|
| Narayanan & Sajith, 2019 – Pixel Count | SOCOFing | Male: 96.4%, Female: 90.2% ([SSRN](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3471862_code3562725.pdf?abstractid=3444032&mirid=1&utm_source=chatgpt.com)) |
| Giudice et al., 2020 – Inception-v3 multi-task | SOCOFing | Gender: 92.5%, Hand: 97.5%, Finger: 92.1% ([arXiv](https://arxiv.org/abs/2007.04931?utm_source=chatgpt.com)) |
| Qi et al., 2021 – DDC-ResNet Autoencoder | SOCOFing | Avg: 96.5%, Male: 97.5%, Female: 95.5% ([arXiv](https://arxiv.org/abs/2108.08233?utm_source=chatgpt.com)) |
| Alhijaj & Khudeyer, 2023 – EfficientNetB0 + RF | SOCOFing | 99.91% ([IIETA](https://iieta.org/download/file/fid/153118?utm_source=chatgpt.com)) |
| **Our Model – CNN + Random Forest** | SOCOFing | **96–99%** |

---

## 🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, seaborn

---

## 📂 Project Structure
Fingerprint-Gender-Orientation-Classifier/
│── data/ # SOCOFing dataset (not included)
│── notebooks/ # Jupyter notebooks for experiments
│── models/ # Saved CNN features + RF model
│── results/ # Reports and plots
│── src/
│ ├── cnn_feature_extractor.py
│ ├── random_forest_classifier.py
│ ├── evaluate.py
│ ├── preprocessing.py
│ └── utils.py
│── requirements.txt
│── README.md
│── LICENSE


---

## ⚡ Installation
```bash
git clone https://github.com/Gopalbhalani137/fingerprint-gender-orientation-classifier.git
cd Fingerprint-Gender-Orientation-Classifier

