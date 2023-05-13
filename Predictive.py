# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import sklearn
loaded_model = pickle.load((open('C:/Users/Sowndarya/PycharmProjects/Diabetes_Predictor/trained_model.sav','rb')))
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array  we predict for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("You are not diabetic")
else:
    print("You are diabetic")
