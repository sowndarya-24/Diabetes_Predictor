# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:29:47 2023

@author: Sowndarya
"""

import numpy as np
import pickle
import streamlit as sl
import sklearn
loaded_model = pickle.load((open('C:/Users/Sowndarya/Python Programs/trained_model.sav','rb')))

#Function creation
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array  we predict for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "You are not diabetic"
    else:
        return "You are diabetic"
    
def main():
    
    #Title
    sl.title('Diabetes Predictor')
    
    #Getting the input data from the user
    pregnancies = sl.text_input('Number of pregnancies')
    glucose = sl.text_input('Glucose_level')
    bloodPressure = sl.text_input('Blood pressure level')
    skinThickness = sl.text_input('Skin Thickness')
    insulin = sl.text_input('Insulin level')
    bmi = sl.text_input('BMI Level')
    diabetesPedigreeFunction = sl.text_input('Diabetes Pedigree function')
    age = sl.text_input('Age')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if sl.button('Test results'):
        diagnosis = diabetes_prediction([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetesPedigreeFunction,age])
    
    sl.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    
    