import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# st.image('logo.jpg')
st.header('Breast Cancer Project')
st.info('⚠️  Kindly provide the information on the left ')

df = pd.read_csv('breast_cancer.csv')
# st.dataframe(df)

# select your model
choice = st.sidebar.selectbox('Select your algorithm', ('Select','KNN', 'SVM'))
st.subheader('Performing EDA on the dataset')

# feature selection
x = df.drop('benign', axis=1)
y = df['benign']

st.write('Total rows', x.shape[0])
st.write('Total columns', x.shape[1])

#  splitting our data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# model training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if choice == 'KNN':
    st.info('KNN model is selected')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    
    # model evaluation
    y_pred_knn = knn.predict(x_test) 
    acc_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)
    st.write('Accuracy Score of KNN is', acc_knn)

elif choice == 'SVM':
    st.info('SVM model is selected')
    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)
    # accuracy score format to 2 digits  
    acc_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 2)
    st.write('Accuracy Score of SVM is', acc_svm)
    
else:
    st.info('Please select your model')