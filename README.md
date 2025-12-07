# Persian Sentiment Analyzer 🇮🇷🤖

**Fine-tuned GPT-2 for 3-class Persian sentiment analysis**: positive/neutral/negative. This project adapts a generative model for a classification task using a custom classification head.



## 📋 Features
- Fine-tuned `HooshvareLab/gpt2-fa` on custom Persian sentiment dataset
- **3-class classification:** Positive (0), Neutral (1), Negative (2)
- Built-in train/validation split with evaluation metrics
- Interactive command-line tool for single-text sentiment prediction

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD21E?style=flat&logo=huggingface&logoColor=black)

## 🚀 Quick Start

**1. Install Dependencies**
```bash
pip install torch transformers datasets accelerate scikit-learn pandas
```

**2. Train Model**
*(Requires a Persian sentiment dataset in CSV/JSON format)*
```bash
python sentiment_analyzer.py --train
```

**3. Predict Sentiment**
```bash
python sentiment_analyzer.py --predict "این کتابخانه خیلی ساکت و آرام است"
# Output: positive
```

## 🎯 Example Usage

```text
Input: "این کتابخانه خیلی مفید و خوب است"
Output: positive

Input: "هوا خیلی آلوده است"
Output: negative
```

## 📁 Files
```text
├── sentiment_analyzer.py  # Fine-tuning + inference pipeline
└── requirements.txt       # Dependencies
```

## 🎓 Model Architecture
- **Base Model:** GPT-2 (HooshvareLab/gpt2-fa)
- **Head:** Sequence Classification (num_labels=3)
- **Batch Size:** 1 (optimized for limited VRAM)
- **Epochs:** 3
- **Optimizer:** AdamW with linear scheduler
- **Parameters:** Warmup steps=500, Weight decay=0.01
