 ## Import des librairies
import streamlit as st
import altair as alt              # for data visualtization
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import pandas as pd
import numpy as np
import os
from zipfile import ZipFile, BadZipFile
import joblib
import pickle
import math
import json
#import Pillow
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')

#API_URL = 'https://dashtest.streamlit.app/'

# Loading the Treated Data
zip_file_path = ZipFile('data_train.zip')
data_train = pd.read_csv(zip_file_path.open('data_train.csv'))

zip_file_test = ZipFile('data_test.zip')
data_test = pd.read_csv(zip_file_test.open('data_test.csv'))

# Loading the model
model = pickle.load(open('model1.pkl', 'rb'))
#with open('model11.pkl', 'rb') as file:
    #model = pickle.load(file)

cols = data_test.select_dtypes(include=['float64','int64']).columns
data_test_scaled = data_test.copy()
data_test_scaled[cols] = StandardScaler().fit_transform(data_test[cols])

col = data_train.select_dtypes(include=['float64','int64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[col] = StandardScaler().fit_transform(data_train[col])

explainer = shap.TreeExplainer(model1)

def check_client_id(client_id: int):
    """
    Customer search in the database
    :param: client_id (int)
    :return: message (string).
    """
    if client_id in list(data_test['SK_ID_CURR']):
        return True
    else:
        return False

# Fonctions
def minmax_scale(df, scaler):
    """Preprocessing du dataframe en paramètre avec le scaler renseigné.
    :param: df, scaler (str).
    :return: df_scaled.
    """
    colmns = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    if scaler == 'minmax':
        scal = MinMaxScaler()
    else:
        scal = StandardScaler()

    df_scaled[colmns] = scal.fit_transform(df[colmns])
    return df_scaled

data_train_mm = minmax_scale(data_train, 'minmax')
data_test_mm = minmax_scale(data_test, 'minmax')

def prediction(client_id: int):
    """
    Calculates the probability of default for a client.
    :param client_id: Client ID (int)
    :return: Probability of default (float) and decision (str)
    """
    client_data = data_test_scaled[data_test_scaled['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    #info_client = info_client[model.feature_names_]
    prediction_proba = model.predict_proba(info_client)
    #prediction = model.predict_proba(info_client)[0][1]
    
    proba_default = round(prediction_proba[:, 1].mean(), 3) if prediction_proba.ndim > 1 else round(prediction_proba[0][1], 3)  
    best_threshold = 0.54
    if proba_default >= best_threshold:
        decision = "Rejected"
    else:
        decision = "Accepted"

    return proba_default, decision


#def get_prediction(client_id):
    """
    Calculates the probability of default for a client.
    :param: client_id (int)
    :return: probability of default (float).
    """
    #client_data = data_test[data_test['SK_ID_CURR'] == client_id]
    #info_client = client_data.drop('SK_ID_CURR', axis=1)
   # prediction = model.predict(info_client)
    #proba = model.predict_proba(info_client)
    #if (prediction == 0) | (prediction == 1):
        #res = '{ "target":'+str(int(prediction))+', "risk":%.2f }'%tuple(proba[0])[1]
    #else:
        #res = 'Erreur du programme!'
    #prediction = model.predict_proba(info_client)[0][1]
    #return prediction

def jauge_score(proba):
    """Constructs a gauge indicating the client's score.
    :param: proba (float).
    """
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=0.54*100,
        #value=proba*100,
        mode="gauge+number+delta",
        title={'text': "Score Gauge"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}}))

    st.plotly_chart(fig)

#def data_voisins(client_id: int):
def df_voisins(client_id: int):
    """Calculates the nearest neighbors of the client_id and returns the dataframe of these neighbors.
    :param client_id: Client ID (int)
    :return: Dataframe of similar clients (DataFrame).
    """
    features = list(data_train_scaled.columns)
    features.remove('SK_ID_CURR')
    features.remove('TARGET')

    # Creating an instance of NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')

    # Training the model on the data
    nn.fit(data_train_scaled[features])
    
    reference_id = client_id
    #reference_observation = data_train_scaled[data_train_scaled['SK_ID_CURR'] == reference_id][features].values
    reference_observation = data_test_scaled[data_test_scaled['SK_ID_CURR'] == reference_id][features].values
    
    if isinstance(reference_observation, np.ndarray):
        reference_df = pd.DataFrame(reference_observation)
        if reference_df.empty or reference_df.isna().any().any():
            return pd.DataFrame()  # Return an empty DataFrame
    else:
        # Handle other types or raise an error if necessary
        pass
    
    #Check if reference_observation is empty or contains NaN values
    #if reference_observation.empty or reference_observation.isna().any().any():
    #if len(reference_observation) == 0 or any([np.isnan(x) for x in reference_observation]):
        #return pd.DataFrame()  # Return an empty DataFrame
    
    # Find nearest neighbors only if reference_observation is not empty
    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = data_train.iloc[indices[0], :]
    
    return df_voisins

#def shap_values_local(id_client: int):
    """ Calcul les shap values pour un client.
        :param: client_id (int)
        :return: shap values du client (json).
        """
    #client_data = data_test_scaled[data_test_scaled['SK_ID_CURR'] == id_client]
    #client_data = client_data.drop('SK_ID_CURR', axis=1)
    #shap_val = explainer(client_data)[0][:, 1]
    ##shap_val = compute_shap_values(id_client_dash)
    #return {'shap_values': shap_val.values.tolist(),
     #       'base_value': shap_val.base_values,
      #      'data': client_data.values.tolist(),
       #     'feature_names': client_data.columns.tolist()}

def shap_values_local(client_id: int):
    """Calculate the SHAP values for a client.
    :param client_id: Client ID (int)
    :return: SHAP values for the client (dict)
    """
    client_data = data_test_scaled[data_test_scaled['SK_ID_CURR'] == client_id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(client_data)
    
    # Construct the output dictionary
    shap_values_dict = {
        'shap_values': shap_values.tolist(),
        'base_value': explainer.expected_value,
        'data': client_data.values.tolist(),
        'feature_names': client_data.columns.tolist()
    }

    return shap_values_dict

def shap_values(explainer, data_scaled):
    """Calculate the SHAP values for the entire dataset.
    :param explainer: SHAP explainer object
    :param data_scaled: Scaled dataset (DataFrame)
    :return: SHAP values as a dictionary
    """
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return {
        'shap_values_0': shap_val[0].tolist(),
        'shap_values_1': shap_val[1].tolist()
    }

def shap_globales(shap_val_glob_0, shap_val_glob_1):
    """Combine and return the global SHAP values.
    :param shap_val_glob_0: SHAP values for class 0 (list)
    :param shap_val_glob_1: SHAP values for class 1 (list)
    :return: Combined SHAP values as a NumPy array
    """
    shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])
    return shap_globales
    
#def shap_val():
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
   # explainer = shap.TreeExplainer(model)
   # shap_val = explainer.shap_values(data_test_scaled.drop('SK_ID_CURR', axis=1))
   # return {'shap_values_0_list': shap_val[0].tolist(),
    #        'shap_values_1': shap_val[1].tolist()}

def distribution(feature, id_client, df):
    """Affiche la distribution de la feature indiquée en paramètre et ce pour les 2 target.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    :param: feature (str), id_client (int), df.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='accordé')
    ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='refusé')

    observation_value = data_test.loc[data_test['SK_ID_CURR'] == id_client][feature].values
    ax.axvline(observation_value, color='green', linestyle='dashed', linewidth=2, label='Client')

    ax.set_xlabel('Valeur de la feature', fontsize=20)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=20)
    ax.set_title(f'Histogramme de la feature "{feature}" pour les cibles accordé et refusé', fontsize=22)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    st.pyplot(fig)

def scatter(id_client, feature_x, feature_y, df):
    """Affiche le nuage de points de la feature_y en focntion de la feature_x.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    :param: id_client (int), feature_x (str), feature_y (str), df.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data_accord = df[df['TARGET'] == 0]
    data_refus = df[df['TARGET'] == 1]
    ax.scatter(data_accord[feature_x], data_accord[feature_y], color='blue',
               alpha=0.5, label='accordé')
    ax.scatter(data_refus[feature_x], data_refus[feature_y], color='red',
               alpha=0.5, label='refusé')

    #data_client = df.loc[df['SK_ID_CURR'] == id_client]
    data_client = data_test.loc[data_test['SK_ID_CURR'] == id_client]
    observation_x = data_client[feature_x]
    observation_y = data_client[feature_y]
    ax.scatter(observation_x, observation_y, marker='*', s=200, color='black', label='Client')

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Analyse bivariée des caractéristiques sélectionnées')
    ax.legend()

    st.pyplot(fig)

def boxplot_graph(id_client, feat, df_vois):
    """Affiche les boxplot des variables renseignéees en paramètre pour chaque target.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    Affiche les 10 plus proches voisins du client sur les boxplot.
    :param: id_client (int), feat (str), df_vois.
    """
    df_box = data_train_mm.melt(id_vars=['TARGET'], value_vars=feat,
                                var_name="variables", value_name="values")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_box, x='variables', y='values', hue='TARGET', ax=ax)


    df_voisins_scaled = minmax_scale(df_vois, 'minmax')
    df_voisins_box = df_voisins_scaled.melt(id_vars=['TARGET'], value_vars=feat,
                                            var_name="var", value_name="val")
    sns.swarmplot(data=df_voisins_box, x='var', y='val', hue='TARGET', size=8,
                  palette=['green', 'red'], ax=ax)

    data_client = data_test_mm.loc[data_test['SK_ID_CURR'] == id_client][feat]
    categories = ax.get_xticks()
    for cat in categories:
        plt.scatter(cat, data_client.iloc[:, cat], marker='*', s=250, color='blueviolet', label='Client')

    ax.set_title(f'Boxplot des caractéristiques sélectionnées')
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 8:
        ax.legend(handles[:4], ['Accordé', 'Refusé', 'Voisins', 'Client'])
    else:
        ax.legend(handles[:5], ['Accordé', 'Refusé', 'Voisins (accordés)', 'Voisins (refusés)', 'Client'])

    st.pyplot(fig)



# Titre de la page
st.set_page_config(page_title="Dashboard Prêt à dépenser", layout="wide")

# Sidebar
with st.sidebar:
    logo = Image.open('logo.png')
    st.image(logo, width=200)
    # Page selection
    page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale",
                                               "Interprétation globale"])

    # ID Selection
    st.markdown("""---""")

    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    #client_id = st.selectbox("ID Client", list_id_client)
    id_client_dash = st.selectbox("ID Client", list_id_client)
    st.write('Vous avez choisi le client ID : '+str(id_client_dash))

    st.markdown("""---""")
    st.write("Created by ...............")

if page == "Home":
    st.title("Dashboard Prêt à dépenser - Home Page")
    st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                "d'approbation ou refus de leur demande de crédit.\n"
                
                "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine). "
                "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                "Lors du déploiement, un échantillon de ces données a été utilisé.\n"
                
                "\nLe dashboard est composé de plusieurs pages :"
                "- **Information du client**: Vous pouvez y retrouver toutes les informations relatives au client "
                "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de crédit. "
                "Je vous invite à accéder à cette page afin de commencer.\n"
                "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéritiques du client ont le plus "
                "influençé le choix d'approbation ou refus de la demande de crédit.\n"
                "- **Intérprétation globale**: Vous pouvez y retrouver notamment des comparaisons du client avec "
                "les autres clients de la base de données ainsi qu'avec des clients similaires.")


if page == "Information du client":
    st.title("Dashboard Prêt à dépenser - Page Information du client")

    st.write("Cliquez sur le bouton ci-dessous pour commencer l'analyse de la demande :")
    button_start = st.button("Statut de la demande")
    if button_start:
        if id_client_dash != '<Select>':
            # Calcul des prédictions et affichage des résultats
            st.markdown("RÉSULTAT DE LA DEMANDE")

            # Call the function and assign the return value to a single variable
            probability, decision = prediction(id_client_dash)

            if probability is not None and decision is not None:
                st.write(f"Probability of Default: {probability}")
                st.write(f"Decision: {decision}")

             # Affichage de la jauge
            jauge_score(proba)
            
            
            #result = get_prediction(client_id)

            # Check if the result is a tuple with two values (probability and decision)
            #if isinstance(result, tuple) and len(result) == 2:
                #probability, decision = result
            #else:
                # Handle the case where only one value is returned
                #probability = result
                #decision = None  # Or handle this case as appropriate for your application
            
            #probability, decision = get_prediction(client_id)
            #probability, decision = get_prediction(id_client_dash)       

            #if decision == 'Accordé':
                #"st.success("Crédit accordé")
            #else:
                #st.error("Crédit refusé")

            # Affichage de la jauge
            #jauge_score(probability)

    # Affichage des informations client
    with st.expander("Afficher les informations du client", expanded=False):
        st.info("Voici les informations du client:", icon='ℹ️')
        st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))
        #st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == client_id]))


if page == "Interprétation locale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation locale")

    locale = st.checkbox("Interprétation locale")
    if locale:
        st.info("Interprétation locale de la prédiction")
        shap_val = shap_val_local(id_client_dash)
        #shap_val = shap_val()     
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)

        if shap_val is not None and len(shap_val) == nb_features:
        #if shap_val is not None and shap_val.shape[0] == nb_features:
        # Affichage du waterfall plot : shap local
            fig = shap.waterfall_plot(shap_val, max_display=nb_features, show=False)
            if fig:
                st.pyplot(fig)
            else:
                st.error("Erreur lors de la création du waterfall plot. Veuillez vérifier vos données.")
        else:
             st.error("Erreur lors du calcul des valeurs SHAP locales. Veuillez vérifier vos données d'entrée.")   
    
        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ici sont affichées les caractéristiques influençant de manière locale la décision. "
                       "C'est-à-dire que ce sont les caractéristiques qui ont influençé la décision pour ce client "
                       "en particulier.")


if page == "Interprétation globale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation globale")
    # Création du dataframe de voisins similaires
    data_voisin = df_voisins(id_client_dash)

    globale = st.checkbox("Importance globale")
    if globale:
        st.info("Importance globale")
        shap_values = shap_val()
        data_test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
        fig, ax = plt.subplots()
        # Affichage du summary plot : shap global
        ax = shap.summary_plot(shap_values[1], data_test_std, plot_type='bar', max_display=nb_features)
        st.pyplot(fig)

        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ici sont affichées les caractéristiques influençant de manière globale la décision.")

    distrib = st.checkbox("Comparaison des distributions")
    if distrib:
        st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données")
        # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
        distrib_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='distrib')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        # Affichage des distributions des variables renseignées
        with st.spinner(text="Chargement des graphiques..."):
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('AMT_CREDIT'))
                if distrib_compa == 'Tous':
                    distribution(feature1, id_client_dash, data_train)
                    #distribution(feature1, client_id, data_train)
                else:
                    #distribution(feature1, client_id, data_voisins)
                    distribution(feature1, id_client_dash, data_voisins)
            with col2:
                feature2 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('EXT_SOURCE_2'))
                if distrib_compa == 'Tous':
                    #distribution(feature2, client_id, data_train)
                    distribution(feature2, id_client_dash, data_train)
                else:
                    #distribution(feature2, client_id, data_voisins)
                    distribution(feature2, id_client_dash, data_voisins)

            with st.expander("Explication des distributions", expanded=False):
                st.caption("Vous pouvez sélectionner la caractéristique dont vous souhaitez observer la distribution. "
                           "En bleu est affichée la distribution des clients qui ne sont pas considérés en défaut et "
                           "dont le prêt est donc jugé comme accordé. En orange, à l'inverse, est affichée la "
                           "distribution des clients considérés comme faisant défaut et dont le prêt leur est refusé. "
                           "La ligne pointillée verte indique où se situe le client par rapport aux autres clients.")

    bivar = st.checkbox("Analyse bivariée")
    if bivar:
        st.info("Analyse bivariée")
        # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
        bivar_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='bivar')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        list_features.insert(0, '<Select>')

        # Selection des features à afficher
        c1, c2 = st.columns(2)
        with c1:
            feat1 = st.selectbox("Sélectionner une caractéristique X ", list_features)
        with c2:
            feat2 = st.selectbox("Sélectionner une caractéristique Y", list_features)
        # Affichage des nuages de points de la feature 2 en fonction de la feature 1
        if (feat1 != '<Select>') & (feat2 != '<Select>'):
            if bivar_compa == 'Tous':
                #scatter(client_id, feat1, feat2, data_train)
                scatter(id_client_dash, feat1, feat2, data_train)
            else:
                #scatter(client_id, feat1, feat2, data_voisins)
                scatter(id_client_dash, feat1, feat2, data_voisins)
            with st.expander("Explication des scatter plot", expanded=False):
                st.caption("Vous pouvez ici afficher une caractéristique en fonction d'une autre. "
                           "En bleu sont indiqués les clients ne faisant pas défaut et dont le prêt est jugé comme "
                           "accordé. En rouge, sont indiqués les clients faisant défaut et dont le prêt est jugé "
                           "comme refusé. L'étoile noire correspond au client et permet donc de le situer par rapport "
                           "à la base de données clients.")

    boxplot = st.checkbox("Analyse des boxplot")
    if boxplot:
        st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données à l'aide de boxplot.")

        feat_quanti = data_train.select_dtypes(['float64']).columns
        # Selection des features à afficher
        features = st.multiselect("Sélectionnez les caractéristiques à visualiser: ",
                                  sorted(feat_quanti),
                                  default=['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

        # Affichage des boxplot
        boxplot_graph(id_client_dash, features, data_voisins)
        #boxplot_graph(id_client_dash, features, data_voisins)
        with st.expander("Explication des boxplot", expanded=False):
            st.caption("Les boxplot permettent d'observer les distributions des variables renseignées. "
                       "Une étoile violette représente le client. Ses plus proches voisins sont également "
                       "renseignés sous forme de points de couleurs (rouge pour ceux étant qualifiés comme "
                       "étant en défaut et vert pour les autres).")
