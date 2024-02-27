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

# Loading the Treated Data
zip_file_path = ZipFile('data_train1.zip')
data_train = pd.read_csv(zip_file_path.open('data_train1.csv'))

zip_file_test = ZipFile('data_test.zip')
data_test = pd.read_csv(zip_file_test.open('data_test.csv'))
featss = [c for c in data_test.columns if c not in ['TARGET','SK_ID_CURR']]

#zip_file = ZipFile('data_selected1.zip')
#data = pd.read_csv(zip_file.open('data_selected1.csv'))
#feats = [c for c in data.columns if c not in ['TARGET','SK_ID_CURR']]

# Loading the model
model = pickle.load(open('model11.pkl', 'rb'))

explainer = shap.TreeExplainer(model)

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
	"""
 	Perform min-max scaling on the DataFrame using the provided scaler.
  	Parameters:
   	df (DataFrame): The DataFrame to be scaled.
	scaler: The scaler object to be used for scaling (e.g., MinMaxScaler).
 	Returns:
  	DataFrame: The scaled DataFrame.
   	"""
	cols = df.select_dtypes(['float64','int32','int64']).columns
	df_scaled = df.copy()
	if scaler == 'minmax':
		scal = MinMaxScaler()  # Corrected class name
	else:
		scal = StandardScaler()
	df_scaled[cols] = scal.fit_transform(df[cols])
	
	return df_scaled

	data_train_mm = minmax_scale(data_train, 'minmax')
	data_test_mm = minmax_scale(data_test, 'minmax')

#def prediction(client_id):
def prediction(client_id:int, data_test, model):
	"""
 	Calculates the probability of default for a client.
  	:param client_id: Client ID (int)
   	:return: Probability of default (float) and decision (str)
	"""
	try :
		client = data_test[data_test['SK_ID_CURR']== client_id]
		if client.empty:
			print("Client not found.")
			return None
		
		# Extract features for the client
		client_features = client.drop('SK_ID_CURR', axis=1)
		proba_default = model.predict_proba(client_features)[0][1]
		#proba_default = round(probab[:, 1].mean(), 3) if probab.ndim > 1 else round(probab[0][1], 3)
		best_threshold = 0.54
		if proba_default >= best_threshold:
			decision = "Rejected"
		else:
			decision = "Accepted"
		
		return proba_default, decision
	
	except Exception as e:
		print("An error occurred during prediction:", e)
		return None, None
		
def jauge_score(prob):
	"""Constructs a gauge indicating the client's score.
 	:param proba: Probability of default (float).
  	"""
	if prob is not None:
		# Define colors for different score ranges
		color_ranges = [(0, 20, "Green"), (20, 45, "LimeGreen"), (45, 54, "Orange"), (54, 100, "Red")]
	
        # Create steps for the gauge based on color_ranges
		steps = []
		for range_start, range_end, color in color_ranges:
			steps.append({'range': [range_start, range_end], 'color': color})
		
		# Define the layout of the gauge
		layout = go.Layout(
			title='Score Gauge',
			margin=dict(l=20, r=20, t=40, b=20),
			xaxis=dict(visible=False),
			yaxis=dict(visible=False),
			paper_bgcolor='rgba(0,0,0,0)',
			plot_bgcolor='rgba(0,0,0,0)'
		)
		
		# Create the figure with the indicator
		fig = go.Figure(go.Indicator(
			domain={'x': [0, 1], 'y': [0, 1]},
			value=proba * 100,
			mode="gauge+number+delta",
			title={'text': "Score Gauge"},
			delta={'reference': 54},
			gauge={
				'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
				'bar': {'color': "MidnightBlue"},
				'steps': steps,
				'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}
			}
		), layout=layout)
		
		# Display the gauge
		st.plotly_chart(fig)
	else:
		st.write("Unable to construct gauge: Probability value is None.")

#def data_voisins(client_id: int):
def df_voisins(client_id: int, data_train, data_test):
	"""
 	Calculates the nearest neighbors of the client_id and returns the dataframe of these neighbors
  	:param client_id: Client ID (int)
   	:return: Dataframe of similar clients (DataFrame).
	"""
	try:
		features = list(data_train.columns)
		features.remove('SK_ID_CURR')
		features.remove('TARGET')

    		# Creating an instance of NearestNeighbors
		knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
 		
		# Training the model on the data
		knn.fit(data_train[features])
		
		reference_id = client_id
		reference_observation = data_test[data_test['SK_ID_CURR'] == reference_id][features].values

		# Check if feature names used during fitting match feature names of query data
		if set(features) != set(data_test.columns):
			raise ValueError("Feature names used during fitting do not match feature names of query data.")
			
    		# Find nearest neighbors only if reference_observation is not empty
		indices = knn.kneighbors(reference_observation, return_distance=False)
		data_voisins = data_train.iloc[indices[0], :]
		
		return data_voisins
	except Exception as e:
        	print("An error occurred during nearest neighbors computation:", e)
        	return None


def shap_values_local(client_id: int, explainer):
	"""Calculate the SHAP values for a client.
 	:param client_id: Client ID (int)
    	:param explainer: SHAP explainer object
    	:return: SHAP values for the client (dict)"""
	
	try:
		# Assuming data_test is accessible inside the function
		client_data = data_test[data_test['SK_ID_CURR'] == client_id]
		client_data = client_data.drop('SK_ID_CURR', axis=1)
        
 	       	# Compute SHAP values
		shap_val = explainer.shap_values(client_data)[0]
        
        	# Construct the output dictionary
		shap_values_dict = {
			'shap_values': shap_val.tolist(),
			'base_value': explainer.expected_value,
			'data': client_data.values.tolist(),
			'feature_names': client_data.columns.tolist()
		}
        
		# Create an Explanation object for further analysis or visualization if needed
		explanation = shap.Explanation(
			values=shap_val,
			base_values=explainer.expected_value,
			data=client_data.values,
			feature_names=client_data.columns
		)
		
		return shap_values_dict
	except Exception as e:
		print("An error occurred during SHAP computation:", e)
		return None

def shap_globales(shap_val_glob_0, shap_val_glob_1):
	"""Combine and return the global SHAP values.
 	:param shap_val_glob_0: SHAP values for class 0 (list)
  	:param shap_val_glob_1: SHAP values for class 1 (list)
   	:return: Combined SHAP values as a NumPy array
	"""

	shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])
	return shap_globales
		
def distribution(feature, id_client, df):
	"""Affiche la distribution de la feature indiquée en paramètre et ce pour les 2 target.
 	Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
  	:param: feature (str), id_client (int), df,
   	"""
	fig, ax = plt.subplots(figsize=(15, 10))
	ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='accordé')
	ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='refusé')
	
	observation_value = data_test.loc[data_test['SK_ID_CURR'] == id_client][feature].values
	ax.axvline(observation_value, color='green', linestyle='dashed', linewidth=2, label='Client')
	
	ax.set_xlabel('Feature value', fontsize=20)
	ax.set_ylabel('Number of occured', fontsize=20)
	ax.set_title(f'Feature Histogram "{feature}" for approved and refused', fontsize=22)
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
	ax.set_title(f'Bivaraiate analysis of selected characteristics')
	ax.legend()
	
	st.pyplot(fig)

def boxplot_graph(id_client, feat, df_vois):
    """
    Displays boxplots of the specified variables for each target class.
    Also displays the position of the client specified by id_client in this graph.
    Displays the 10 nearest neighbors of the client on the boxplots.
    :param id_client: Client ID (int)
    :param feat: List of features to be plotted (list of str)
    :param df_vois: DataFrame containing the nearest neighbors data
    """
    try:
        # Melt the DataFrame for boxplot visualization
        df_box = df_vois.melt(id_vars=['TARGET'], value_vars=feat,
                              var_name="variables", value_name="values")

        # Create boxplot
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(data=df_box, x='variables', y='values', hue='TARGET', ax=ax)

        # Scale the nearest neighbors data
        df_voisins_scaled = minmax_scale(df_vois[feat])

        # Melt the scaled DataFrame for swarmplot visualization
        df_voisins_box = df_voisins_scaled.melt(id_vars=['TARGET'], value_vars=feat,
                                                var_name="var", value_name="val")

        # Create swarmplot for nearest neighbors
        sns.swarmplot(data=df_voisins_box, x='var', y='val', hue='TARGET', size=8,
                      palette=['green', 'red'], ax=ax)

        # Get the data for the specified client
        data_client = df_vois[df_vois['SK_ID_CURR'] == id_client][feat]

        # Highlight the client on the plot
        for feat in feat:
            ax.scatter(feat, data_client[feat], marker='*', s=250, color='blueviolet', label='Client')

        # Set plot title and legend
        ax.set_title('Boxplot des caractéristiques sélectionnées', fontsize=22)
        ax.legend(fontsize=12, loc='best')

        st.pyplot(fig)

    except Exception as e:
        print("An error occurred:", e)
	    


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
    #id_client = st.selectbox("ID Client", list_id_client)
    id_client_dash = st.selectbox("ID Client", list_id_client)
    #st.write('Vous avez choisi le client ID : '+str(id_client))
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
			proba_default, decision = prediction(id_client_dash, data_test, model)
			if decision is None:  # Check if only one value is returned (error message)
				st.write(proba_default)  # Display the error message
			else:  # Two values returned (probability and decision)
				st.write(f"Probability of Default: {proba_default}")
				st.write(f"Decision: {decision}")
			
				# Affichage de la jauge
				if proba_default is not None:
					jauge_score(proba_default)
				else:
					st.write("Unable to construct gauge: Probability value is None.")
				
# Affichage des informations client
with st.expander("Afficher les informations du client", expanded=False):
	client_info = data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]
	if not client_info.empty:
		st.info("Voici les informations du client:", icon='ℹ️')
		st.write(pd.DataFrame(client_info))
	else:
		st.warning("Aucune information trouvée pour le client avec l'ID spécifié.")

if page == "Interprétation locale":
	st.title("Dashboard Prêt à dépenser - Page Interprétation locale")
	
	locale = st.checkbox("Interprétation locale")
	if locale:
		st.info("Interprétation locale de la prédiction")
		shap_val = shap_values_local(id_client_dash, explainer)

		if shap_val is not None and len(shap_val)>0:
			nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
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
	#data_voisin = df_voisins(id_client_dash)

	globale = st.checkbox("Importance globale")
	if globale:
		st.info("Importance globale")
		shap_values = shap_values_local(id_client_dash, explainer)
		data_test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
		nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
		fig, ax = plt.subplots()
		
	# Affichage du summary plot : shap global
	ax = shap.summary_plot(shap_values[1].reshape(1, -1), data_test_std, plot_type='bar', max_display=nb_features)
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

    # Function to display distribution for a selected feature
    def display_distribution(feature, data):
        if distrib_compa == 'Tous':
            distribution(feature, id_client_dash, data_train)
        else:
            distribution(feature, id_client_dash, data_voisins)

    # Display distributions for selected features
    with st.spinner(text="Chargement des graphiques..."):
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Choisissez une caractéristique", list_features, index=list_features.index('AMT_CREDIT'))
            display_distribution(feature1, data_train)
        with col2:
            feature2 = st.selectbox("Choisissez une caractéristique", list_features, index=list_features.index('EXT_SOURCE_2'))
            display_distribution(feature2, data_train)

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
				scatter(id_client_dash, feat1, feat2, data_train)
			else:
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
        feat_quanti = data_train.select_dtypes(['float64','int32','int64']).columns
        # Selection des features à afficher
	default_features = ['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
	features = st.multiselect("Sélectionnez les caractéristiques à visualiser:",
			  sorted(feat_quanti), default=default_features)

# Affichage des boxplot
boxplot_graph(id_client_dash, features, data_voisins)
with st.expander("Explication des boxplot", expanded=False):
	st.caption("Les boxplot permettent d'observer les distributions des variables renseignées. "
		   "Une étoile violette représente le client. Ses plus proches voisins sont également "
		   "renseignés sous forme de points de couleurs (rouge pour ceux étant qualifiés comme "
		   "étant en défaut et vert pour les autres).")
