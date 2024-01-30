 ## Import des librairies
import math
import shap
import streamlit as st
import altair as alt              # for data visualtization
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import requests
import plotly
import os
from zipfile import ZipFile, BadZipFile
import pickle


# Features
feat = ['SK_ID_CURR','TARGET','DAYS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN','DAYS_EMPLOYED','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']

# Nombre de ligne
num_rows = 100000

# Original Data
zip_file_path = 'sample_application_train.zip'
try:
    with ZipFile(zip_file_path, 'r') as zip_file:
       raw_train = pd.read_csv(zip_file.open('sample_application_train.csv'), usecols=feat, nrows=num_rows) 
except BadZipFile:
    print(f"Error: '{zip_file_path}' is not a valid ZIP file.")
except Exception as e:
    print(f'An unexpected error occured: {e}')

try:
    raw_test = pd.read_csv(zip_file_test.open('application_test.csv'),usecols=[f for f in feat if f!='TARGET'], nrows=num_rows)
except Exception as e:
    print(f'Error reading test data:{e}')        

try:
    raw_app = raw_train.append(raw_test).reset_index(drop=True)    # Append the DataFrames
    #raw_app = pd.concat([raw_train, raw_test], ignore_index=True)

except Exception as e:
    # Print the exception message for debugging
    print(f"Error concatenating DataFrames: {e}")

