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

#st.title("💳💵 Credit Score Dashboard")

#st.write("💳💵 Crédit Score Dashboard")
##################################################

# Page configuration initiatlisation
st.set_page_config(
  page_title="Credit Score Dashboard", 
  page_icon="💳💵",
  layout="wide",
  #initial_sidebar_state="expanded",
  #menu_items=None)
alt.themes.enable("dark")

# Sidebar
  with st.sidebar:
    logo_path = "logo.png"
    try:
      logo = Image.open(logo_path)
      st.image(logo, width=150)
    except FileNotFoundError:
      st.error(f"Error: Logo file not found at {logo_path}")

  #logo_path = "api_7/logo.png"
  #logo = Image.open(logo_path)
  #st.image(logo, width=180)
  #logo = "💵"
  #logo = Image.open("api_7/pret à dépenser.png")
  #st.image(logo, width=200)

  # Page selection
page =  st.selectbox("Navigation", ["Home", "Client Information", "Local interpretation", "Global interpretation"])
  
  #Id selection
st.markdown("""---""")
  
  #list_id_client = list(data_test['SK_ID_CURR'])
  #list_id_client.insert(0, '<Select>')
  #id_client_dash = st.selectbox("ID Client", list_id_client)
  #st.write('Vous avez choisi le client ID : '+str(id_client_dash))

st.markdown("""---""")
st.write("By Amit GAUTAM")
