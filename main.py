import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv')
print(df.head()) #First five rows

print(df.shape) #Number of rows and columns
print(df.describe())

#Separating data and labels

x = df.drop(columns = 'Outcome' , axis = 1) #Axis=1 means, dropping a column
y = df['Outcome']

#Data standardization
#scaler = StandardScaler()
#scaler.fit(x)
#sd = scaler.transform(x)
#x = sd

#Splitting Training and test data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#Training the model
classifier = svm.SVC(kernel='linear')

#Training the support vector machine classifier

classifier.fit(x_train,y_train)

#Model Evaluation
#Accuracy score
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

#Making a predective system
input_data = (10,115,0,0,0,35.5,0,134.8)
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array  we predict for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#Standardizing the input data
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("You are not diabetic")
else:
    print("You are diabetic")

#Saving the trained model
import pickle
filename = 'trained_model.sav'
pickle.dump(classifier,open(filename, 'wb'))

#Loading the saved model
loaded_model = pickle.load((open('trained_model.sav','rb')))


