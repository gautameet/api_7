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
# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

# Page configuration inistiatlisation
#st.set_page_config(page_title="Credit Score Dashboard", page_icon="💳💵", layout="wide", initial_sidebar_state="expanded", menu_items=None)
#alt.themes.enable("dark")


##################################################
