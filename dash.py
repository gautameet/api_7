## Import des librairies
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
import plotly
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
#import json
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pickle

##################################################

# Page configuration initialisation
st.set_page_config(
  page_title="Credit Score Dashboard", 
  page_icon="💳💵",
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items=None)

alt.themes.enable("dark")

# Sidebar
with st.sidebar:
    logo_path = "logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=180)
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {logo_path}")
      
    # Page selection
    page =  st.selectbox("Navigation", ["Home", "Customer Information", "Local interpretation", "Global interpretation"])
  
    #Id selection
    #st.markdown("""---""")
  
    #st.markdown("""---""")
    
    #st.write("By Amit GAUTAM")

if page == "Home":
    st.title("💳💵 Credit Score Dashboard - Home Page")
    ".\n"
    #.\n"
  
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer Information**: to find out all the information related to the customer.\n"
                "- **Local Interpretation**: Information regarding the reasons for accepting or refusing the credits of a particular customer.\n"
                "- **Global Interpretation**: Information regarding the comparisons and similarity between the customer according to the database.")

if page == "Customer Information":
    st.title("💳💵 Credit Score Dashboard - Customer Information - Page")

    st.write("To analyse your demand, please click the button below :")
    button_start = st.button("Demand Status")
    if button_start:
        if id_client_dash != '<Select>':
            # Calculates the prediction and displays the results"
            st.markdown("Result of your request")
            probability, decision = get_prediction(id_client_dash)

            if decision == 'Approved':
                st.success("Loan approved")
            else:
                st.error("Loan refused")

          
            # Affichage de la jauge
            jauge_score(probability)
          
   # Display customer information
    with st.expander("Display customer information", expanded=False):
        st.info("The customer information are:")
        #st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))
