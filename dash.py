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

#alt.themes.enable("dark")

# Sidebar
with st.sidebar:
    logo_path = "logo.png"
    try:
      logo = Image.open(logo_path)
      st.image(logo, width=150)
    except FileNotFoundError:
      st.error(f"Error: Logo file not found at {logo_path}")


# Page selection
page =  st.selectbox("Navigation", ["Home", "Customer Information", "Local interpretation", "Global interpretation"])
  
#Id selection
st.markdown("""---""")
  
#list_id_client = list(data_test['SK_ID_CURR'])
#list_id_client.insert(0, '<Select>')
#id_client_dash = st.selectbox("ID Client", list_id_client)
#st.write('Vous avez choisi le client ID : '+str(id_client_dash))

st.markdown("""---""")
st.write("By Amit GAUTAM")

if page == "Home":
    st.title("💳💵 Credit Score Dashboard - Home Page")
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused.\n"
                
                "\nThis automatique predictions are calculted with the help of machine learning algorith, "
                                
                "\nThis dashboard is composed of following pages :\n"
                "- **Client Information**: to find out all the information related to the customer.\n"
                "- **Local Interpretation**: Information regarding the reasons for accepting or refusing the credits of a particular customer.\n"
                "- **Global Interpretation**: Information regarding the comparisons and similarity between the customer according to the database.")

