import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os





model=joblib.load('best_model.pkl')
pipeline=joblib.load('pipeline.pkl')
input=pd.read_csv('input1.csv')
input.drop(['Survived'],axis=1,inplace=True)
   
transform_input=pipeline.transform(input)
predictions=model.predict(transform_input)

input['predicted_survived']=predictions

input.to_csv('titanic_output.csv',index=False)
print("Inference complete. Results saved to titanic_output.csv")
    


