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
    page_icon="💳💵",
    layout="wide",
    initial_sidebar_state="expanded")
