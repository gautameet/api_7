# Import des librairies
import streamlit as st
from PIL import Image
import requests
#import plotly.graph_objects as go
#import plotly.express as px
import plotly.figure_factory as ff
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##################################################

# Page configuration inistiatlisation

st.set_page_config(
    page_titile="Credit Score Dashboard",
    page_icon="💳💵",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enables("dark")


##################################################
