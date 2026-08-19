# 🎬 IMDB Movie Reviews Sentiment Analysis & Text Mining

A comprehensive End-to-End Natural Language Processing (NLP) and Machine Learning project focused on binary sentiment classification (Positive vs. Negative) of IMDB movie reviews using Python, NLTK, and Scikit-Learn.

---

## 📌 Project Overview

This repository demonstrates a complete data science pipeline for Sentiment Analysis on the benchmark **IMDB Dataset** (50,000 movie reviews). The primary objective is to classify text reviews into **Positive** (0) or **Negative** (1) sentiments using classical Machine Learning models.

### Key Highlights:
- **Dataset Size:** 50,000 balanced reviews (25,000 positive, 25,000 negative).
- **Text Preprocessing:** Automated HTML tag removal, non-alphabetic character filtering, lowercasing, tokenization, and stop-word filtering using NLTK.
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization.
- **Modeling & Comparison:** Logistic Regression vs. Naive Bayes (MultinomialNB).
- **Exploratory Text Analysis:** N-gram/Word frequency analysis & data visualization using Seaborn and Matplotlib.

---

## 📊 Dataset Information

- **Source:** IMDB Dataset of 50K Movie Reviews
- **Columns:**
  - `review` *(string)*: Raw text review written by users.
  - `sentiment` *(string/categorical)*: Target label (`positive` / `negative`).

| Metric | Details |
| :--- | :--- |
| **Total Rows** | 50,000 |
| **Class Distribution** | 25,000 Positive / 25,000 Negative (100% Balanced) |
| **Train/Test Split** | 80% Train (40,000 samples) / 20% Test (10,000 samples) |

---

## 🛠️ Project Pipeline

1. **Exploratory Data Analysis (EDA):**
   - Inspection of data shapes, data types, missing values, and class balance.
   - Mapping string labels (`positive` $\rightarrow$ `0`, `negative` $\rightarrow$ `1`).

2. **Text Preprocessing & Cleaning:**
   - **Lowercasing:** Converting text to lower case for uniformity.
   - **HTML Tag Removal:** Removing regex patterns like `<br />` / `<br>`.
   - **Punctuation & Digit Removal:** Keeping only English alphabetic characters (`[a-zA-Z]`).
   - **Tokenization:** Splitting reviews into individual tokens via `nltk.tokenize.word_tokenize`.
   - **Stop Words Removal:** Filtering out common English stop words using `nltk.corpus.stopwords`.

3. **Feature Extraction (TF-IDF):**
   - Transformed cleaned text reviews into numerical sparse matrices using `TfidfVectorizer()`.
   - Applied `fit_transform` on the training set and `transform` on the test set to prevent data leakage.

4. **Model Training & Evaluation:**
   - **Logistic Regression:** Primary classifier trained and evaluated with accuracy and classification report.
   - **Multinomial Naive Bayes (Bonus Model):** Alternative probabilistic classifier for performance benchmarking.

5. **Text Analytics & Visualization:**
   - Extraction of Top 10 most frequent words in positive and negative reviews using `Counter`.
   - Visualizing word frequencies using comparative Seaborn bar charts (`Greens_r` for positive, `Reds_r` for negative).

---

## 📈 Model Performance & Results

Both models were tested on the held-out test dataset (10,000 reviews):

| Model | Accuracy | F1-Score (Positive) | F1-Score (Negative) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **89.81%** | **0.90** | **0.90** | 🏆 **Best Performer** |
| **Multinomial Naive Bayes** | 86.69% | 0.87 | 0.86 | Baseline Benchmark |

### Detailed Classification Report (Logistic Regression):
```text
               precision    recall  f1-score   support

    0 (Pos)       0.89      0.91      0.90      5039
    1 (Neg)       0.91      0.88      0.90      4961

   accuracy                           0.90     10000
  macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
