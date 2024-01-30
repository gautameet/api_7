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
num_rows = 150000

# Original Data
zip_file_path = 'sample_application_train.zip'
try:
    with ZipFile(zip_file_path, 'r') as zip_file:
       raw_train = pd.read_csv(zip_file.open('sample_application_train.csv'), usecols=feat, nrows=num_rows) 
except BadZipFile:
    print(f"Error: '{zip_file_path}' is not a valid ZIP file.")
except Exception as e:
    print(f'An unexpected error occured: {e}')


