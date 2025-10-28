import pandas as pd
import numpy as np
import os 
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scale',StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ('oneimputer',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline=ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data=pd.read_csv('housing.csv')

    data['income_cat']=pd.cut(data['median_income'],
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train ,test in split.split(data,data['income_cat']):
        data_train=data.loc[train].drop(['income_cat'],axis=1)
        data_test=data.loc[test].drop(['income_cat'],axis=1)
    data_test.to_csv('input.csv',index=False)


    labels=data_train['median_house_value'].copy()
    features=data_train.drop(['median_house_value'],axis=1)

    num_attribs=features.drop(['ocean_proximity'],axis=1).columns.tolist()
    cat_attribs=["ocean_proximity"]

    pipeline=build_pipeline(num_attribs,cat_attribs)
    pre_features=pipeline.fit_transform(features)

    model=RandomForestRegressor(random_state=42)
    model.fit(pre_features,labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model trained and saved.")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    input=pd.read_csv('input.csv')
    transform_input=pipeline.transform(input)
    predictions=model.predict(transform_input)
    
    input['median_house_value']=predictions

    input.to_csv("output1.csv",index=False)
    print("Inference complete. Results saved to output.csv")
