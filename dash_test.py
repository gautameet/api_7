## Import des librairies
import shap
import streamlit as st
import altair as alt              # for data visualtization
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
    
    st.header("Welcome to the customers' page.\n")
    ".\n"
    st.subheader("Please enter your ID to know the results of your demands. \n") 
    #"Thank you. \n"
    with st.sidebar:
        st.selectbox("Enter your ID", "      ")
        button_start = st.button("Submit")
    
    
    
                #if button_start:
# Use the entered customer_id to fetch and display relevant information
# Add your logic here
                #customer_info = fetch_customer_info("Navigation")
                #st.write(customer_info)

    #Id selection
    import streamlit as st

# First, create the layout with three columns
col1, col2, col3 = st.columns(3)

# Now, add content to each column
with col1:
    st.header("Column 1")
    st.write("Content for column 1 goes here.")

with col2:
    st.header("Column 2")
    st.write("Content for column 2 goes here.")

with col3:
    st.header("Column 3")
    st.write("Content for column 3 goes here.")
