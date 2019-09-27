from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import glob
import os
import numpy as np

cv = CountVectorizer(stop_words="english", max_features=500)

emails, labels = [], []

file_path = 'E:/Projects/Python Projects/Email Spam/data/enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path = 'E:/Projects/Python Projects/Email Spam/data/enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)
        

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def letters_only(astr):
    return astr.isalpha()

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if word.isalpha()
                                        and word not in all_names]))
    return cleaned_docs

cleaned_emails=clean_text(emails)
cleaned_emails_cv = cv.fit_transform(cleaned_emails)

feature_mapping = cv.vocabulary
feature_names = cv.get_feature_names()

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

email_test = [
    '''Subject: flat screens
    hello ,
    please call or contact regarding the other flat screens requested .
    trisha tlapek - eb 3132 b
    michael sergeev - eb 3132 a
    also the sun blocker that was taken away from eb 3131 a .
    trisha should two monitors also michael .
    thanks
    kevin moore''',
    '''Subject: having problems in bed ? we can help !
    cialis allows men to enjoy a fully normal sex life without having to plan the sexual act .
    if we let things terrify us , life will not be worth living .
    brevity is the soul of lingerie .
    suspicion always haunts the guilty mind .''',
]

cleaned_test=[]
for doc in email_test:
    cleaned_test.append(' '.join([lemmatizer.lemmatize(word.lower())
                                    for word in doc.split()
                                    if word.isalpha()
                                    and word not in all_names]))
    
email_test_cv = cv.transform(cleaned_test)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)

X_train_cv= cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train_cv, y_train)
prediction_prob = clf.predict_proba(X_test_cv)
#prediction_prob[0:10]
prediction = clf.predict(X_test_cv)
prediction[:10]
acc=accuracy_score(prediction,y_test)
print('The accuracy using MultinomialNB is: {0:.1f}%'.format(acc*100))

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, prediction, pos_label=1)
recall_score(y_test, prediction, pos_label=1)
f1_score(y_test, prediction, pos_label=1)

print(12)