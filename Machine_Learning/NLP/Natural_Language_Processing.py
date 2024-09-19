import pandas as pd
import numpy as np 
#%%
data = pd.read_csv(r"gender_classifier_dataset.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender = [1 if i == "female" else 0 for i in data.gender]
#%% data cleaning,regular expression
import re
first_description = data.description[4] 
description = re.sub("[^a-zA-z]"," ",first_description) #Find which one string doesnt equal from a to z and A to Z and change with " "
description = description.lower()
#%% stopwords(irrelavent words)
import nltk 
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords

description = description.split()
#description = nltk.word_tokenize(description)
# if we use tokenizer method instead split method we wont lose datas which is like shouldn't 
#%% remove unnecessary words
description = [word for word in description if not word in set(stopwords.words("english"))]
#%% Lemmatization(finding roots of words)
import nltk as nlp
lemma = nltk.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)
#%%
description_list = []
for i in data.description:
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = description.split()
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    description = [lemma.lemmatize(word) for word in description]
    lemma = nltk.WordNetLemmatizer()
    description = " ".join(description)
    description_list.append(description)
    
#%%
from sklearn.feature_extraction.text import CountVectorizer

max_features = 500

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))
#%%
y = data.iloc[:,0].values   # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

print("accuracy: ",knn.score(prediction.reshape(-1,1),y_test)

#%%
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
















