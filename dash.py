## Import des librairies
#import shap
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from PIL import Image
import requests
import plotly
import os
from zipfile import ZipFile
import pickle
#import xgboost
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
import json
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns



### Import des donnees
# Features
feat = ['SK_ID_CURR','TARGET','DAYS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
        'DAYS_EMPLOYED','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
                
# Nombre de ligne
#num_rows = 150000

zip_file_train = './data_train.zip'
zip_file_test = './data_test.zip'

with ZipFile(zip_file_train, 'r') as zip_train:
    df_train = pd.read_csv(zip_train.open('data_train.csv'))

with ZipFile(zip_file_test, 'r') as zip_test:
    df_test=pd.read_csv(zip_test.open('data_test.csv'))    

# Modele voisin
#knn = NearestNeighbors(n_neighbors=10)
#knn.fit(df_train.drop(['SK_ID_CURR','TARGET'], axis=1))


#Importing model 
#pkl_model= open("./model.pkl","rb")
#model = pickle.load(pkl_model)

#explainer = shap.TreeExplainer(model, df_train)

# Features
#features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']


## Page configuration initialisation
st.set_page_config(
    page_title="Credit Score Dashboard",
    page_icon="💳💵",
    layout="wide",
    initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    logo_path = "logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=180)
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {logo_path}")
      
# Page selection
 page =  st.selectbox("Navigation", ["Home", "Customer", "Local interpretation", "Global interpretation"])
  
#Id selection
st.markdown("""---""")

customer_id_list = list(df_test['SK_ID_CURR'])
customer_id_list.insert(0, '<Select>')
customer_id_dash = st.selectbox("Customer_id", customer_id_list)
if customer_id_dash != '<Select>':
     st.write(f'You have chosen the Customer ID: {customer_id_dash}')
else:
     st.info("Please select a customer ID.")
                            
     st.markdown("""---""")
    
     st.write("By Amit GAUTAM")
                   
    if page == "Home":
        st.title("💳💵 Credit Score Dashboard - Home Page")
        ".\n"
        #.\n"
    
        st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                    "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                
                    "\nThis dashboard is composed of following pages :\n"
                    "- **Customer**: to find out all the information related to the customer.\n"
                    "- **Local Interpretation**: Information regarding the reasons for accepting or refusing the credits of a particular customer.\n"
                    "- **Global Interpretation**: Information regarding the comparisons and similarity between the customer according to the database.")
    
    if page == "Customer":
        st.title("💳💵 Credit Score Dashboard - Customer")
    
    #Display customer information
        ".\n"
        st.title("Welcome to the customer page")
    
        ".\n"
        st.write("Please click the below button to enter:")
    
        button_start = st.button("Your ID number")
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
                st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))

    if page == "Local interpretation":
        st.title("💳💵 Credit Score Dashboard - Local Interpretation - Page")

        locale = st.checkbox("Local Interpretation")
        if locale:
            st.info("Local Interpretation prediction")
            #shap_val = get_shap_val_local(id_client_dash)
            nb_features = st.slider("Number of variables to display", 0, 20, 10)
        
        # Diplaying waterfall plot : shap local
            fig = shap.waterfall_plot(shap_val, max_display=nb_features, show=False)
            st.pyplot(fig)

            with st.expander("Graphical presentation", expanded=False):
                st.caption("Displaying the features that influence the decision locally (for a particular customer)")

    if page == "Global interpretation":
        st.title("💳💵 Credit Score Dashboard - Global interpretation - Page")
   
        # Création du dataframe de voisins similaires
        data_voisins = df_voisins(id_client_dash)

        globale = st.checkbox("Global Importance")
        if globale:
            st.info("Global Importance")
            shap_values = get_shap_val()
            data_test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
            nb_features = st.slider("Number of varaibles to display", 0, 20, 10)
            fig, ax = plt.subplots()
       
            # Displaying summary plot : shap global
            ax = shap.summary_plot(shap_values[1], data_test_std, plot_type='bar', max_display=nb_features)
            st.pyplot(fig)
    
            with st.expander("Graphical presentation", expanded=False):
                st.caption("Displaying the features that influence the decision in global scenario.")

        distrib = st.checkbox("Distribution comparision")
        if distrib:
            st.info("Disptribution comparaison wtih other variables from the data")
          
            #Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
            distrib_compa = st.radio("Choose the type of comparision :", ('All', 'Similar clients'), key='distrib')
    
            list_features = list(data_train.columns)
            list_features.remove('SK_ID_CURR')
           
            #Affichage des distributions des variables renseignées
            with st.spinner(text="Charging graphs..."):
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox("Please choose one feature", list_features, index=list_features.index('AMT_CREDIT'))
                    if distrib_compa == 'All':
                        distribution(feature1, id_client_dash, df_train)
                    else:
                        distribution(feature1, id_client_dash, data_voisins)
                with col2:
                    feature2 = st.selectbox("Please choose one feature", list_features, index=list_features.index('EXT_SOURCE_2'))
                    if distrib_compa == 'All':
                        distribution(feature2, id_client_dash, df_train)
                    else:
                        distribution(feature2, id_client_dash, data_voisins)

                with st.expander("Distribution explanation", expanded=False):
                    st.caption("You can select the feature that you like to observe the distribution."
                               "Blue shows the clients distribution not having credit default and their "
                               "loan is considered to be approved (Loan). Orange shows the clients distribution "
                               "having credit default and their loan is considered to be refused. "
                               "The green dotted line indicates where the customer stands in relation to other customers."
                              )
        bivar = st.checkbox("Bivariate Analysis")
        if bivar:
            st.info("Bivariate Analysis")
            # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
            bivar_compa = st.radio("Please choose a comprarision type :", ('All', 'Similar Clients'), key='bivar')
            
            list_features = list(df_train.columns)
            list_features.remove('SK_ID_CURR')
            list_features.insert(0, '<Select>')
            
            # Selection des features à afficher
            c1, c2 = st.columns(2)
            with c1:
                feat1 = st.selectbox("Select a feature X ", list_features)
            with c2:
                feat2 = st.selectbox("Select one feature Y", list_features)
                # Affichage des nuages de points de la feature 2 en fonction de la feature 1
            if (feat1 != '<Select>') & (feat2 != '<Select>'):
                if bivar_compa == 'All':
                    scatter(customer_id_dash, feat1, feat2, data_train)
                else:
                    scatter(customer_id_dash, feat1, feat2, data_voisins)
                with st.expander("Explaining the scatter plot", expanded=False):
                    st.caption("Vous pouvez ici afficher une caractéristique en fonction d'une autre. "
                               "En bleu sont indiqués les clients ne faisant pas défaut et dont le prêt est jugé comme "
                               "accordé. En rouge, sont indiqués les clients faisant défaut et dont le prêt est jugé "
                               "comme refusé. L'étoile noire correspond au client et permet donc de le situer par rapport "
                               "à la base de données clients.")
        boxplot = st.checkbox("Boxplot analysis")
        if boxplot:
            st.info("Comparing the distribution of many variables of the total data from the boxplot.")
            
            feat_quanti = data_train.select_dtypes(['float64']).columns
            # Selection des features à afficher
            features = st.multiselect("Sélectionnez les caractéristiques à visualiser: ",
                                      sorted(feat_quanti),
                                      default=['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])
            # Affichage des boxplot
            boxplot_graph(id_client_dash, features, data_voisins)
                
        with st.expander("Explaining box plot", expanded=True):
            st.write('''
                "The boxplot permits an observer on the distribution of known variable.",
                "A star viol one star violet which represent a customer.",
                "Its nearest neighbour also undertaken on colour form red for qualified by default but green not."''')
 
