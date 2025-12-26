# Fake News Detection

This project focuses on detecting fake news articles using Natural Language Processing (NLP) techniques.  
Text data is cleaned and converted into numerical form using the TF-IDF Vectorizer, followed by training machine learning models to classify news as Fake or Real

---

##  Dataset

Source: Kaggle – Fake and Real News Dataset  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Files:
- True.csv → Real news (label = 1)  
- Fake.csv → Fake news (label = 0)

Columns:
- title, text, subject, date

---

## Technologies Used
- Python
- Numpy, Pandas
- Matplotlib, Seaborn 
- Scikit-learn
- TF-IDF Vectorizer
- Jupyter Notebook

---

## 🔄 Project Workflow

1. Load and explore dataset  
2. Clean and preprocess text data  
   - Removed Reuters prefixes  
   - Converted to lowercase  
   - Removed extra spaces  
3. Assign labels and merge datasets  
4. Shuffle data and split into train/test sets  
5. Convert text into TF-IDF features  
6. Train models  
   - Logistic Regression  
   - Multinomial Naive Bayes  
7. Evaluate using:
   - Classification report  
   - Confusion matrix  
8. Interpret feature weights and analyze misclassified samples

---

## What I Learned
- TF-IDF captures writing patterns, not factual truth
- Dataset bias strongly affects learned features
- Logistic Regression assigns weights to words
- Stopword removal changes what the model learns.
- Naive Bayes relies on independent word probabilities and is less robust

---
