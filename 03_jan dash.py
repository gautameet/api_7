import streamlit as st
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap  

st.set_page_config(page_title="Prédiction de la capacité de remboursement d'un demandeur de prêt",
                   page_icon="🏦",
                   layout="wide",
                   initial_sidebar_state="expanded")

with st.container():
    st.title("Prédiction de la capacité de remboursement d'un demandeur de prêt")
    st.markdown("❗*Cet outil permet d'assister à la prise de décision et doit être utilisé conjointement avec une analyse approfondie réalisée par un professionel*❗")
    st.markdown('##')

req_i = requests.post("https://k4jzhwndavohdljtjjpxwp.streamlit.app/id_client")
resultat_i = req_i.json()

st.sidebar.markdown("Selection du client")
option = st.sidebar.selectbox("Veuillez spécifier le numéro d'identification du demandeur de prêt",(resultat_i["list_id"]))

    
if st.button("Prediction"):

    schema = {"num_client": option, "feat":"string"}
        
    req = requests.post("https://k4jzhwndavohdljtjjpxwp.streamlit.app/perso_info", json=schema)
    resultat = req.json()
    if resultat["gender"] == 0:
        st.sidebar.write(f"Genre:   Female")
    else:
        st.sidebar.write(f"Genre:   Male")
    st.sidebar.write(f"Situation familiale:   {resultat['family']}")
    st.sidebar.write(f"Nombre d'enfants:   {resultat['nb_child']}")
    st.sidebar.write(f"Montant du crédit demandé:   {round(resultat['credit']):,}")
    st.sidebar.write(f"Revenu:   {round(resultat['income_amount']):,}")
    st.sidebar.write(f"Source du revenu:   {resultat['income_type']}")

                
    req1 = requests.post("https://k4jzhwndavohdljtjjpxwp.streamlit.app/predict", json=schema)
    resultat1 = req1.json()
    st.write(resultat1["verdict"])
    st.write(resultat1["proba"])
            
    req2 = requests.post("https://k4jzhwndavohdljtjjpxwp.streamlit.app/gauge", json=schema)
    resultat2 = req2.json()
    st.components.v1.html(resultat2["fig"], height=500)


    req3 = requests.post("https://k4jzhwndavohdljtjjpxwp.streamlit.app/explanation", json=schema)
    resultat3 = req3.json()
    st.dataframe(resultat3["df_feat"])  

    st.components.v1.html(resultat3["fig"], height=50
