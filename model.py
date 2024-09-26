import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model():
    heart_disease = pd.read_csv('heart_disease_data.csv')
    X = heart_disease.drop(columns='target', axis=1)
    Y = heart_disease['target']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    return model

model = train_model()

def predict_heart_disease(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)
    return prediction[0]
