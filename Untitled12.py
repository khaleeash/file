#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sns.set_style('darkgrid')

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            df_review = pd.read_csv(file_path)
            process_data(df_review)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def process_data(df_review):
    df_positive = df_review[df_review['sentiment']=='positive'][:9000]
    df_negative = df_review[df_review['sentiment']=='negative'][:1000]
    df_review_imb = pd.concat([df_positive, df_negative])

    colors = sns.color_palette('deep')

    plt.figure(figsize=(8,4), tight_layout=True)
    plt.bar(x=['Positive', 'Negative'],
            height=df_review_imb.value_counts(['sentiment']),
            color=colors[:2])
    plt.title('Sentiment')
    plt.show()

    rus = RandomUnderSampler(random_state=0)
    df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

    train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

    train_x, train_y = train['review'], train['sentiment']
    test_x, test_y = test['review'], test['sentiment']

    tfidf = TfidfVectorizer(stop_words='english')
    train_x_vector = tfidf.fit_transform(train_x)
    test_x_vector = tfidf.transform(test_x)

    svc = SVC(kernel='linear')
    svc.fit(train_x_vector, train_y)

    score = svc.score(test_x_vector, test_y)
    f1_scores = f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)
    classification_rep = classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])
    conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])

    messagebox.showinfo("Results", f"Accuracy: {score}\nF1 Scores: {f1_scores}\nClassification Report:\n{classification_rep}\nConfusion Matrix:\n{conf_mat}")

root = tk.Tk()
root.title("Sentiment Analysis GUI")

open_button = tk.Button(root, text="Open File", command=open_file)
open_button.pack(pady=20)

root.mainloop()


# In[ ]:




