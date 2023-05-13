# Diabetes_Predictor
A mini  Machine learning project that predicts whether a person has diabetes based on the input that has been fed.

**Workflow of the project**

The dataset used here is from the UCI Machine learning repository, Kaggle. The basic predictor model is built in main.py using the PyCharm IDE for training and testing the datasets.
The next step, is to install Anaconda Navigator and create a new environment 'Machine Learning'. 
Install Qtconsole and Spyder to the environment.
Open the terminal of Machine learning environment and execute the following:
  pip install numpy
  pip install pandas
  pip install scikit-learn
  pip install streamlit

The spyder editor is used to deploy the Machine learning project using the Streamlit framework.(Can use Pycharm also).
Streamlit is used for the UI to accept inputs from the users.
In the spyder IDE, two files are created predictive.py and predictive_app.py
In predictive.py the model is loaded from the main.py and in the predictive_app.py the streamlit framework is imported to design the UI for the project.

After completing the project, open the terminal of the Machine Learning environment and execute the following command:
streamlit run "path_of_your_predictive_app.py"

