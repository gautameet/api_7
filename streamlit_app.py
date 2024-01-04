# Import des librairies
import streamlit as st
from PIL import Image
import requests
#import plotly.graph_objects as go
#import plotly.express as px
#import plotly.figure_factory as ff
#import json
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

st.title("💳💵 Credit Score Dashboard")

#st.write("💳💵 Crédit Score Dashboard")
##################################################

# Page configuration inistiatlisation
st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
#st.set_page_config(page_title="Credit Score Dashboard", page_icon="💳💵", layout="wide", initial_sidebar_state="expanded", menu_items=None)
#alt.themes.enable("dark")


##################################################
