import re
import string
import nltk
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedKFold
from sklearn.utils import shuffle
nltk.download('stopwords')
from nltk.corpus import stopwords
import pyarabic.araby as araby

stopwords_list = stopwords.words('arabic')
stopwords_list2 = ['الا', 'ان', '']
data_set = pd.read_csv('Data_set.csv')
shuffle(data_set)
x = data_set['tweets']
y = data_set['label']

test = pd.DataFrame(x)

test.columns = ["tweets"]

# remove puctuation
def remove_punct(txt):
    txt_res = "".join([c for c in txt if c not in string.punctuation])
    return txt_res
test['tweets'] = test['tweets'].apply(lambda x: remove_punct(x))
#/////////////////////////////////////
x = test['tweets']
# remove stop words
test['tweets'] = test['tweets'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in ((stopwords_list) and (stopwords_list2))]))
x = test['tweets']
# remove emojis
test['tweets'] = test['tweets'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
x = test['tweets']
# remove tashkeel
test['tweets'] = test['tweets'].apply(araby.strip_diacritics)
x = test['tweets']
# //////////////////////////////////////////////////////////////////////////

# transform tweets from string to numeric value for (x)
vect = TfidfVectorizer()
x = vect.fit_transform(x)
# encode to 0,1 for (y)
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)
# cross validation v
kf = RepeatedKFold(n_splits=20, n_repeats=2, random_state=1)
for train_index, test_index in kf.split(data_set):
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
# logistic
#  small datasets, ‘liblinear’ is a good choice,
model = LogisticRegression(solver='liblinear', C=10, random_state=0)
print("Results of of LogisticRegression:")

model.fit(x_train, y_train)
p = model.predict(x_test)

# true positive /true positive +false positive &minimize false positive
print("precision of LogisticRegression", precision_score(y_test, p))
#  true positive / actual positive =(tp+fn)
print("recall of LogisticRegression", recall_score(y_test, p))

# acc=(tn+tp)/n n=dataset
print("accuracy of LogisticRegression", accuracy_score(y_test, p))
#2 * (precision * recall) / (precision + recall)
print("F-measure of LogisticRegression", f1_score(y_test, p, average='binary'))
print("//////////////////////////////////////////////////////////////////")
print("results of svm:")
# two classifications only (pos& neg)+binary classification
clf = svm.SVC(kernel='linear')
# train the model
clf.fit(x_train, y_train)
# testing and predicting by using the testing data set x_test and y_test
y_pred = clf.predict(x_test)
# //////////////////////////////////////////////////////////////////////////////
print("precision of svm", precision_score(y_test, y_pred))
print("recall of svm", recall_score(y_test, y_pred))
print("F-measure of svm", f1_score(y_test, y_pred, average='binary'))
print("accuracy of svm", accuracy_score(y_test, y_pred))
