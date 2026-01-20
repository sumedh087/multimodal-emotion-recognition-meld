# Multimodal Emotion Recognition on MELD  
### Whisper + BERT Late Fusion Model (120M Parameters)

This repository contains the full implementation of a multimodal emotion recognition system using **Whisper audio embeddings** and **BERT text embeddings**, combined through a **lightweight late‑fusion neural architecture**. The project achieves:

- **62% Accuracy**
- **62% Weighted F1**
- On the **MELD** (Multimodal EmotionLines) dataset

This work emphasizes **efficiency**, **reproducibility**, and **deployability**, achieving strong performance with *half the parameters* of recent state‑of‑the‑art models.

---
## 🔥 Key Features
- **Audio Modeling:** Whisper encoder + convolutional projection  
- **Text Modeling:** Fine‑tuned BERT (CLS token embedding)  
- **Fusion Strategy:** Late concatenation (832‑dim vector → FC layers)  
- **Balanced Loss:** Class‑weighted CrossEntropy  
- **Evaluation:** Confusion matrices, weighted F1, class‑wise metrics  
- **Reproducibility:** Fixed seeds, environment files included  

---
## 📁 Repository Structure
```
emotion-recognition-multimodal/
│
├── src/  
│   ├── audio_model/  
│   ├── text_model/  
│   ├── fusion_model/  
│   └── train.py  
│
├── notebooks/
│   └── experiments.ipynb
│
├── reports/
│   ├── AML_Project.pdf
│   └── script.ipynb
│
├── data/
│   ├── README.md  (data download instructions)
│
├── models/
│   ├── bert/
│   ├── whisper/
│   └── fusion/
│
├── environment.yml  
├── requirements.txt  
├── .gitignore
└── README.md
```

---
## 🗂 Dataset: MELD
- Download: https://github.com/declare-lab/MELD  
- Place files in: `data/meld/`  
- Requires: audio clips, text transcripts, YAML label files

---
## 🧪 Training
```
python src/train.py --config configs/train_fusion.yaml
```

---
## 📊 Results (Test Set – 2,609 samples)
| Emotion | F1 Score |
|--------|----------|
| Neutral | **0.78** |
| Joy | 0.56 |
| Surprise | 0.55 |
| Anger | 0.46 |
| Sadness | 0.37 |
| Fear | 0.23 |
| Disgust | 0.23 |

**Overall F1:** **0.62**  
**Accuracy:** **62%**

---
## 📄 Reports  
Full technical documentation is available in:  
- `reports/AML_Project.pdf`  
- `reports/1768922993_script.pdf`

---
## 🛠 Environment Setup
```
conda env create -f environment.yml
conda activate emotion-env
```
OR pip:
```
pip install -r requirements.txt
```

---
## 📧 Contact
Project contributors:  
- **Sumedh Galgali**  
- **Ritika Pandey**
