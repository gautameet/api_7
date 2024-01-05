# Import des librairies
import streamlit as st
from sklearn.preprocessing import StandardScaler
#from PIL import Image
#import requests
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

st.title("💳💵 Credit Score Dashboard")

st.write("💳💵 Crédit Score Dashboard")
##################################################

# Page configuration inistiatlisation
st.set_page_config(page_title="Credit Score Dashboard", page_icon="💳💵", layout="wide", initial_sidebar_state="expanded", menu_items=None)
alt.themes.enable("dark")

# Sidebar
with st.sidebar:
  logo = Image.open("img/logo pret à dépenser.png")
  st.image(logo, width=200)

  # Page selection
  page =  st.selectbox("Navigation", ["Home", "Client Information", "Local interpretation", "Global interpretation"])
  
  #Id selection
  st.markdown("""---""")
  
  list_id_client = list(data_test['SK_ID_CURR'])
  list_id_client.insert(0, '<Select>')
  id_client_dash = st.selectbox("ID Client", list_id_client)
  st.write('Vous avez choisi le client ID : '+str(id_client_dash))

  st.markdown("""---""")
  st.write("By Amit GAUTAM")
