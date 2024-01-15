## Import des librairies
#import shap
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests
import plotly
import os
from zipfile import ZipFile
import pickle
#import xgboost
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
import json
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


zip_file_train = './data_train.zip'
zip_file_test = './data_test.zip'

with ZipFile(zip_file_train, 'r') as zip_train:
    df_train = pd.read_csv(zip_train.open('data_train.csv'))

with ZipFile(zip_file_test, 'r') as zip_test:
    df_test=pd.read_csv(zip_test.open('data_test.csv'))    

#Importing model 
pkl_model= open("./model.pkl","rb")
model = pickle.load(pkl_model)

#explainer = shap.TreeExplainer(model, df_train)


        
########################
# DASHBOARD

## Page configuration initialisation
st.set_page_config(
    page_title="Credit Score Dashboard",
    page_icon="💳💵",
    layout="wide",
    initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    logo_path = "logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=180)
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {logo_path}")
      
# Page selection
    page =  st.selectbox("Navigation", ["Home", "Customer"])   #,"Local interpretation", #"Global interpretation"])
  
#Id selection
    st.markdown("""---""")
 
    
    st.markdown("""---""")
    
    #st.write("By Amit GAUTAM")
                   
if page == "Home":
    st.title("💳💵 Credit Score Dashboard - Home Page")
    ".\n"
    #.\n"
    
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer**: to find out all the information related to the customer.\n")
                #"- **Local Interpretation**: Information regarding the reasons for accepting or refusing the credits of a particular customer.\n"
                #"- **Global Interpretation**: Information regarding the comparisons and similarity between the customer according to the database.")
    
if page == "Customer":
    st.title("💳💵 Credit Score Dashboard - Customer")
    
#Display customer information
    ".\n"
    st.title("Welcome to the customer page")
    ".\n"    
    
    
    st.write("Please insert you ID:")
    button_start = st.button("Submit")
    
    st.markdown("""---""")

    st.write("By Amit GAUTAM")
        #customer_id = st.sidebar.number_input('Enter Customer ID:', min_value=1, max_value=df_train['Customer_ID'].max())
    
  
