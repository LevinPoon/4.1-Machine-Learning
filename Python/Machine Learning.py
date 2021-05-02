#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.externals
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

size = float(input("Please input the test size?"))

music_data = pd.read_csv("C:/Users/Hp/Desktop/Codes/15 Machine Learning Supervised Learning/Python/Music.csv")

x_train = music_data.drop(columns = ['Genre'])
y_train = music_data['Genre']

x_train, x_test, y_train, y_test = train_test_split (x_train, y_train, test_size = size)
model = DecisionTreeClassifier()

model.fit(x_train,y_train)

predictions = model.predict(x_test)
score = accuracy_score (y_test, predictions)

percentage = "{:.2%}".format(score)
scoreword = str(percentage)

print("The accuracy score is: " + percentage + "\n")
print("1. Predict the genre for 22 male?")
predictions1 = model.predict ([[22,1]])
print("The answer is: " + predictions1)
print("")
print("2. Predict the genre for 29 female?")
predictions2 = model.predict ([[29,0]])
print("The answer is: " + predictions2)
print("")
print(music_data)


# In[ ]:




