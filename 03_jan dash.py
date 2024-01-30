 ## Import des librairies
import math
import shap
import streamlit as st
import altair as alt              # for data visualtization
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import requests
import plotly
import os
from zipfile import ZipFile, BadZipFile
import pickle

