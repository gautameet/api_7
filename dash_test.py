
## Import des librairies
import streamlit as st
import altair as alt              # for data visualtization
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import FixedLocator
import requests
import plotly
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

# Features
feat = ['SK_ID_CURR','TARGET','DAYS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
        'DAYS_EMPLOYED','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']

# Nombre de ligne
num_rows = 150000

# Original Data
zip_file_train = ZipFile('sample_application_train.zip')
raw_train = pd.read_csv(zip_file_train.open('sample_application_train.csv'),usecols=feat, nrows=num_rows)
zip_file_test = ZipFile('application_test.zip')
raw_test = pd.read_csv(zip_file_test.open('application_test.csv'),usecols=[f for f in feat if f!='TARGET'])

raw_app = pd.concat([raw_train, raw_test], ignore_index=True)        #concat
del raw_train
del raw_test
 
raw_app.loc[:, 'YEARS_EMPLOYED'] = raw_app['DAYS_EMPLOYED'].apply(lambda x: -x/-365)
raw_app.loc[:, 'AGE'] = raw_app['DAYS_BIRTH'].apply(lambda x: -x/-365) // (-365)
raw_app.loc[:, 'CREDIT'] = raw_app['AMT_CREDIT'].apply(lambda x: 'No' if math.isnan(x) else 'Yes')       

# Drop unnecessary columns
raw_app = raw_app.drop(['DAYS_BIRTH','DAYS_EMPLOYED'], axis=1).copy()

# Treated Data
zip_file_path = ZipFile('data_train.zip')
train = pd.read_csv(zip_file_path.open('data_train.csv'))

zip_file_test = ZipFile('data_test.zip')
test = pd.read_csv(zip_file_test.open('data_test.csv'))

#Concat
app = pd.concat([train, test], ignore_index=True)
 
# Nearest neighbors
knn = NearestNeighbors(n_neighbors=10)
knn.fit(train.drop(['SK_ID_CURR','TARGET'], axis=1))

# Loading the model
with open('model11.pkl', 'rb') as file:
    model = pickle.load(file)

# Explainer
zip_file = ZipFile('X_train_sm_split1.zip')
X_train_sm_1 = pd.read_csv(zip_file.open('X_train_sm_split1.csv'))

zip_file = ZipFile('X_train_sm_split2.zip')
X_train_sm_2 = pd.read_csv(zip_file.open('X_train_sm_split2.csv'))

zip_file = ZipFile('X_train_sm_split3.zip')
X_train_sm_3 = pd.read_csv(zip_file.open('X_train_sm_split3.csv'))

X_train_sm = pd.concat([X_train_sm_1, X_train_sm_2, X_train_sm_3], ignore_index=True)
X_train_sm.reset_index(drop=True, inplace=True)       # Reset the index to have a continuous index for the concatenated DataFrame
X_name = list(X_train_sm.columns)

np.bool = np.bool_
np.int = np.int_

# Shap Explainer
explainer = shap.TreeExplainer(model, X_train_sm)

del X_train_sm_1
del X_train_sm_2
del X_train_sm_3
del X_train_sm

# Features
features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']

# Data
def get_data(data,ID):
    if type(ID) == list:
        return data.loc[data['SK_ID_CURR'].isin(ID)]
    else:
        return data.loc[data['SK_ID_CURR']==ID].head(1)


# Neighbor
def get_similar_ID(ID):
    app_id = app.loc[app['SK_ID_CURR']==ID].drop(['SK_ID_CURR','TARGET'], axis=1)
    knn_index = knn.kneighbors(app_id,return_distance=False)
    knn_id = app['SK_ID_CURR'][app.index.isin(knn_index[0])].values.tolist()
    return knn_id

def get_stat_ID(ID):   
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna()
    return len(data_knn),len(data_knn[data_knn['TARGET']==1])

## GRAPH
# Graph radar initialisation

def _invert(x, limits):
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    scaled_data = []

    # Extract the first range
    x1, x2 = ranges[0]
    # Check if the range is inverted
    if x1 > x2:
        x1, x2 = x2, x1  # Swap values if inverted

    # Scale each data point to fit within the range
    for d, (y1, y2) in zip(data, ranges):
        # Check if the range is inverted
        if y1 > y2:
            y1, y2 = y2, y1  # Swap values if inverted
        # Scale the data point to fit within the range
        scaled_value = (d - y1) / (y2 - y1) * (x2 - x1) + x1
        scaled_data.append(scaled_value)

    return scaled_data

#def _scale_data(data, ranges):
    #for d, (y1, y2) in zip(data[1:], ranges[1:]):
        #assert (y1 <= d <= y2) or (y2 <= d <= y1)
    #x1, x2 = ranges[0]
    #d = data[0]
    #if x1 > x2:
        #d = _invert(d, (x1, x2))
       # x1, x2 = x2, x1
    #sdata = [d]
   # for d, (y1, y2) in zip(data[1:], ranges[1:]):
     #   if y1 > y2:
      #      d = _invert(d, (y1, y2))
      #      y1, y2 = y2, y1
      #  sdata.append((d-y1) / (y2-y1) 
      #                   * (x2 - x1) + x1)
  #  return sdata

class ComplexRadar():
    def __init__(self,fig,variables,ranges,n_ordinate_levels=6):
        angles = np.arange(0,360,360./len(variables))
    
        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,label="axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,labels=variables)
        [txt.set_rotation(angle-90) for txt,angle in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) for x in grid]
            if ranges[i][0]>ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid,labels=gridlabel,angle=angles[i])
            ax.set_ylim(*ranges[i])
            ax.set_yticklabels(ax.get_yticklabels(),fontsize=6)   # Increased fontsize for y tick labels
            #ax.set_xticklabels(variables,fontsize=6)    # Adjusted fontsize for x tick labels
        
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax.set_xticklabels(variables,fontsize=6)
    
    def plot(self,data,*args,**kw):
        scaled_data=_scale_data(data,self.ranges)      # Ensure _scale_data function works correctly
        self.ax.plot(self.angle,np.r_[scaled_data,scaled_data[0]],*args, **kw)        
    
    def fill(self,data,*args,**kw):
        scaled_data=_scale_data(data,self.ranges)
        self.ax.fill(self.angle,np.r_[scaled_data, scaled_data[0]],*args,**kw)
        
# Graph Radar
def radat_id_plot(ID,fig, features=features,fill=False):
    app_id = get_data(raw_app,ID).loc[:,features]
    client = app_id.iloc[0]

    ranges = {
        'AGE': (client['AGE'] - 5, client['AGE'] + 5),
        'YEARS_EMPLOYED': (client['YEARS_EMPLOYED'] - 1, client['YEARS_EMPLOYED'] + 1),
        'AMT_INCOME_TOTAL': (client['AMT_INCOME_TOTAL'] - 500, client['AMT_INCOME_TOTAL'] + 500),
        'AMT_ANNUITY': (client['AMT_ANNUITY'] - 100, client['AMT_ANNUITY'] + 100),
        'AMT_CREDIT': (client['AMT_CREDIT'] - 500, client['AMT_CREDIT'] + 500)
    }

    radar = ComplexRadar(fig, features, ranges)
    radar.plot(client, linewidth=3, color='darkseagreen')

    if fill:
        radar.fill(client, alpha=0.2)
    

def radat_knn_plot(ID,fig,features=features,fill=False):
    # Get data for the specified client ID
    app_id = get_data(raw_app,ID).loc[:,features]
    data_id = app_id.iloc[0]    

    # Get similar IDs using KNN
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna().copy()
    
    data_knn['TARGET'] = data_knn['TARGET'].astype(int)
    moy_knn = data_knn.groupby('TARGET').mean()

    ranges = [(min(data_knn['AGE']), max(data_knn['AGE'])),
              (min(data_knn['YEARS_EMPLOYED']),  max(data_knn['YEARS_EMPLOYED'])),
              (min(data_knn['AMT_INCOME_TOTAL']),  max(data_knn['AMT_INCOME_TOTAL'])),
              (min(data_knn['AMT_ANNUITY']),  max(data_knn['AMT_ANNUITY'])),
              (min(data_knn['AMT_CREDIT']),  max(data_knn['AMT_CREDIT']))]
    
     # Calculate ranges for radar plot based on data_id
    #ranges = [(data_id['AGE'] - 5, data_id['AGE'] + 5),
              #(data_id['YEARS_EMPLOYED'] - 1, data_id['YEARS_EMPLOYED'] + 1),
              #(data_id['AMT_INCOME_TOTAL'] - 500, data_id['AMT_INCOME_TOTAL'] + 500),
              #(data_id['AMT_ANNUITY'] - 100, data_id['AMT_ANNUITY'] + 100),
              #(data_id['AMT_CREDIT'] - 500, data_id['AMT_CREDIT'] + 500)]
    
    # Perform radar plot using ranges
    radar = ComplexRadar(fig, features, ranges)
    radar.plot(data_id,linewidth=3,label='Client '+str(ID),color='darkseagreen')
    radar.plot(moy_knn.iloc[1][features],linewidth=3,label='Client similaire moyen avec difficultés',color='red')
    radar.plot(moy_knn.iloc[0][features],linewidth=3,label='Client similaire moyen sans difficultés',color='royalblue')
    fig.legend(fontsize=5,loc='upper center',bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    
    # Optionally fill radar plot
    if fill:
        radar.fill(client, alpha=0.2)
    
def shap_id(ID):
    app_id = get_data(app,ID).loc[:, X_name].copy()
    shap_vals = explainer.shap_values(app_id)
    shap_plot = shap.bar_plot(shap_vals[1][0],feature_names=X_name,max_display=10)

    # Return the plot object
    return shap_plot

# Loading data
zip_file = ZipFile('data_selected1.zip')
df_sel = pd.read_csv(zip_file.open('data_selected1.csv'))
feats = [c for c in df_sel.columns if c not in ['TARGET', 'SK_ID_CURR']]

# defining Prediction
def predict_target():
    ID=st.number_input("Enter Client ID:", min_value=100002, max_value=456255)
    
    try:
        ID_data = df_sel.loc[df_sel['SK_ID_CURR'] == ID]
        if ID_data.empty:
            return "Client not found!"
        ID_to_predict = ID_data.loc[:, feats]        #feature of data_selected1

        # Make predictions
        prediction = model.predict(ID_to_predict)
        proba = model.predict_proba(ID_to_predict)[0][1]
        best_threshold = 0.54
        
        if proba >= best_threshold:
            decision = "Approved"
        else:
            decision = "Refused"
        
        return proba, decision
    
    except Exception as e:
        return f"Error occurred: {str(e)}"


###############################################
## DASH BOARD
########################

    ## Page configuration initialisation
#st.set_page_config(page_title="Credit Score Dashboard", page_icon="💵", layout="wide", initial_sidebar_state="expanded")
    
    # Sidebar
with st.sidebar:
    st.write("Credit Score Dashboard")
    logo_path = "logo.png"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=250)
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {logo_path}")
    
# Page selection
page =  st.sidebar.selectbox("Menu", ["Home", "Customer", "Customer portfolio"])
    
            
st.markdown("-----")
    
st.sidebar.write("By: Amit GAUTAM")
    

if page == "Home":
    st.title("💵 Credit Score Dashboard - Customer Page")
    ".\n"
           
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                    
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer**: to find out all the information related to the customer.\n")
                    
if page == "Customer":
    st.title("💵 Welcome to the Customer Page")
    ".\n"
    st.sidebar.markdown("Please select your ID:")
    #st.markdown("Your ID:")
    ID=st.sidebar.number_input(" ", min_value=100002, max_value=456255)
    print("debug", ID)
    raw_app_id = get_data(raw_app,ID)
    with st.spinner('Custumer details....'):
        st.write('## Customer details.....')
        with st.container():
            col1, col2 = st.columns([1.5,2.5])      
            with col1:
                st.write("#### Customer detail " + str(ID))
                st.markdown("* **Status : " + str(raw_app_id['NAME_FAMILY_STATUS'].values[0]) + "**")
                st.markdown("* **Number of children) : " + str(raw_app_id['CNT_CHILDREN'].values[0]) + "**")
                st.markdown("* **Employment: " + str(raw_app_id['NAME_INCOME_TYPE'].values[0]) + "**")
                st.markdown("* **Current Loan : " + str(raw_app_id['CREDIT'].values[0]) + "**")
            
            with col2:
                fig = plt.figure(figsize=(2,2))
                #radat_id_plot(ID,fig,features=features,fill=False)
                st.pyplot(radat_id_plot(ID,fig))
                    
        st.markdown("-----")
        
        with st.container():
            st.write("#### Similar type of Customers ")
            #try:
            col3, col4 = st.columns([3,1])
            with col3:
                fig = plt.figure(figsize=(3,3))
                #radat_knn_plot(ID,fig,features=features)
                st.pyplot(radat_knn_plot(ID,fig))
            with col4:
                N_knn, N_knn1 = get_stat_ID(ID)
                st.markdown("* **Similar type of customers : " + str(N_knn) + "**")
                st.markdown("* **Customer having payment problem : " + str(N_knn1) + "**")                
                st.markdown("_(either " + str(N_knn1*100/N_knn) + "% clients with similar payment problems)_")
                
            
        st.markdown("-----")
        with st.container():
            st.write("#### Customer solvability prediction ")
        prediction_button = st.button('Predict solvability')
        if prediction_button:
            with st.spinner('Calculating...'):
                try:
                    result = predict_target()
                    if isinstance(result, tuple) and len(result) == 2:
                        proba, decision = result
                        if decision=="Approved":                            
                            st.write(':smiley:')
                            st.success(f'Client solvable (Target = 0), prediction difficulty level at **{proba * 100:.2f}%**')
                        elif decision=="Refused":
                            st.write(':confused:')
                            st.error(f'Client non solvable (Target = 1), prediction difficult level at **{proba * 100:.2f}%**')  
                            st.write('**Interpretability**')
                            fig = plt.figure(figsize=(2,2))
                            st.pyplot(shap_id(ID))
                    else:
                        st.warning(result)
                except Exception as e:
                    st.warning(f'Programme error:{str(e)}') 
                    st.write(':dizzy_face:')                                               
    
# Customer portfolio analysis        
if page == 'Customer portfolio':
    st.write("### Customer portfolio analysis")
    with st.spinner('Analysing...'):
        with st.container():            
            st.write("#### Customer Profile")
            col1, col2,col3 = st.columns(3)
            plt.ioff()
            with col1:
                fig, ax = plt.figure(figsize=(8,6))
                bins = int((raw_app['AGE'].max() - raw_app['AGE'].min()) // 5)
                
                sns.histplot(data=raw_app, x='AGE', hue='TARGET', bins=bins, palette=['royalblue', 'red'], alpha=0.5, ax=ax)
                ax.set(xlabel='AGE', ylabel='Frequency')
                ax.legend(['Having difficulty', 'Without difficulty'], loc='lower center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=5)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=raw_app, x='NAME_FAMILY_STATUS', y='CNT_CHILDREN', hue='TARGET', palette=['royalblue', 'red'], alpha=0.5, ci=None, edgecolor='black', ax=ax)
                ax.set_xlabel('Family Status')
                ax.set_ylabel('Number of Children')
                ax.set_title('Number of Children by Family Status and Target')
                ax.legend(title='Target', fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                plt.setp(ax.get_yticklabels(), fontsize=8)
                st.pyplot(fig,ax)

            with col3:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=raw_app, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', hue='TARGET', palette=['royalblue', 'red'], alpha=0.5, ci=None, edgecolor='black', ax=ax)
                ax.set_xlabel('Income Type')
                ax.set_ylabel('Income Total')
                ax.set_title('Income Total by Income Type and Target')
                ax.legend(title='Target', fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                plt.setp(ax.get_yticklabels(), fontsize=8)
                st.pyplot(fig,ax)

                
        st.markdown("-----")
               
        with st.container():
            st.write("#### Loan Payment")
            tg_n = np.array([len(raw_app.loc[raw_app['TARGET']==1]), len(raw_app.loc[raw_app['TARGET']==0]), len(raw_app.loc[raw_app['TARGET'].isnull()])])            
            col4, col5 = st.columns(2)
    
            with col4:
                fig = plt.figure(figsize=(4, 4))
                labels = ['Having difficulty', 'Without difficulty', 'No Loan outstanding']
                colors = ['red', 'royalblue', 'honeydew']
                plt.pie(tg_n, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
                plt.axis('equal')
                st.pyplot(fig)

            with col5:
                df = raw_app[['TARGET', 'NAME_INCOME_TYPE', 'AMT_ANNUITY', 'AMT_CREDIT']]
                df.loc[:, 'COUNT_TG'] = df['TARGET']

                tg_df = pd.concat((df.groupby(['TARGET', 'NAME_INCOME_TYPE']).mean()[['AMT_ANNUITY', 'AMT_CREDIT']],
                                   df.groupby(['TARGET', 'NAME_INCOME_TYPE']).count()[['COUNT_TG']]), axis=1)
                tg_0 = tg_df.loc[0]
                tg_1 = tg_df.loc[1]

                fig = plt.figure(figsize=(4, 4))
                sns.scatter(tg_1['AMT_ANNUITY'], tg_1['AMT_CREDIT'], s=tg_1['COUNT_TG'].values/100, label='With difficulty', color='red')
                sns.scatter(tg_0['AMT_ANNUITY'], tg_0['AMT_CREDIT'], s=tg_0['COUNT_TG'].values/100, label='Without difficulty', color='royalblue', alpha=.3)

                # Customize plot
                plt.legend(loc='upper left', fontsize=8)
                plt.xlabel('AMT_ANNUITY', fontsize=8)
                plt.ylabel('AMT_CREDIT', fontsize=8)
                plt.xlim([20000, 40000])
                plt.ylim([400000, 800000])
                plt.setp(pt.get_xticklabels(),fontsize=6)
                plt.setp(pt.get_yticklabels(),fontsize=6)
                st.pyplot(fig)
                #plt.xticks(fontsize=6)
                #plt.yticks(fontsize=6)
                #st.pyplot(fig)
