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

########################
# DASHBOARD

## Page configuration initialisation
st.set_page_config(
    page_title="Credit Score Dashboard",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.write("Credit Score Dashboard")
    logo_path = "logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=250)
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {logo_path}")

# Page selection
    page =  st.selectbox("Navigation", ["Home", "Customer"])



    
    
    st.markdown("""---""")
    
    st.write("By: Amit GAUTAM")



if page == "Home":
    st.title("💵 Credit Score Dashboard - Home Page")
    ".\n"
    #.\n"
    
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer**: to find out all the information related to the customer.\n")
                
if page == "Customer":
    st.title("💵 Welcome to the Customer Page")
    ".\n"
    #.\n"
    
st.markdown("Welcome to the customers' page. Please enter your ID to know the results of your demands. Thank you. \n"
    with st.sidebar:
    #st.write("Please insert you ID:")
    #customer_id = st.number_input('Enter Customer ID:', min_value=1)
        cust_id = st.selectbox("Enter your ID", "      ")
        button_start = st.button("Submit")
    
#Display customer information
#st.title("Welcome et Bienvenue")
#".\n"    
    
    
    if button_start:
# Use the entered customer_id to fetch and display relevant information
# Add your logic here
        customer_info = fetch_customer_info("Navigation")
        st.write(customer_info)

    #Id selection

    
