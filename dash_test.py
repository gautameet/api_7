## Import des librairies
import streamlit as st
import altair as alt              # for data visualtization
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
num_rows = 100000

# Original Data
zip_file_train = ZipFile('sample_application_train.zip')
raw_train = pd.read_csv(zip_file_train.open('sample_application_train.csv'),usecols=feat, nrows=num_rows)

zip_file_test = ZipFile('./application_test.zip')
raw_test = pd.read_csv(zip_file_test.open('application_test.csv'),usecols=[f for f in feat if f!='TARGET'])

raw_app = pd.concat([raw_train, raw_test], ignore_index=True)        #concat
del raw_train
del raw_test
 
raw_app.loc[:, 'YEARS_EMPLOYED'] = raw_app['DAYS_EMPLOYED'].apply(lambda x: -x/-365)
raw_app.loc[:, 'AGE'] = raw_app['DAYS_BIRTH'].apply(lambda x: -x/-365) // (-365)
#raw_app['CREDIT'] = raw_app['AMT_CREDIT']   
raw_app.loc[:, 'CREDIT'] = raw_app['AMT_CREDIT'].apply(lambda x: 'No' if math.isnan(x) else 'Yes')       

#st.dataframe(raw_app)

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

# Loading data
zip_file = ZipFile('data_selected1.zip')
data = pd.read_csv(zip_file.open('data_selected1.csv'))
feats = [c for c in data.columns if c not in ['TARGET','SK_ID_CURR']]

# Loading the model
with open('model11.pkl', 'rb') as file:
    model = pickle.load(file)
#model = pickle.load(open('model11.pkl','rb'))

# Explainer
zip_file = ZipFile('X_train_sm_split1.zip')
X_train_sm_1 = pd.read_csv(zip_file.open('X_train_sm_split1.csv'))

zip_file = ZipFile('X_train_sm_split2.zip')
X_train_sm_2 = pd.read_csv(zip_file.open('X_train_sm_split2.csv'))

zip_file = ZipFile('X_train_sm_split3.zip')
X_train_sm_3 = pd.read_csv(zip_file.open('X_train_sm_split3.csv'))

#X_train_sm = X_train_sm_1.append(X_train_sm_2).reset_index(drop=True).append(X_train_sm_3).reset_index(drop=True)
X_train_sm = pd.concat([X_train_sm_1, X_train_sm_2, X_train_sm_3], ignore_index=True)
X_train_sm.reset_index(drop=True, inplace=True)       # Reset the index to have a continuous index for the concatenated DataFrame
X_name = list(X_train_sm.columns)

##ADDED TODAY
np.bool = np.bool_
np.int = np.int_
explainer = shap.TreeExplainer(model, X_train_sm)

#explainer = shap.TreeExplainer(model, X_train_sm)

del X_train_sm_1
del X_train_sm_2
del X_train_sm_3
del X_train_sm

# Features
features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']

# Data
def get_data(data, ID):
    if isinstance(ID, list):
        return data[data['SK_ID_CURR'].isin(ID)].copy()
    else:
        return data[data['SK_ID_CURR'] == ID].head(1).copy()
#def get_data(data,ID):
    #if type(ID) == list:
        #return data[data['SK_ID_CURR'].isin(ID)]
    #else:
        #return data[data['SK_ID_CURR']==ID].head(1)

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
    #inverts a value x on a scale from
    #limits[0] to limits[1]
    

def _scale_data(data, ranges):
    #scales data[1:] to ranges[0],
    #inverts if the scale is reversed
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                         * (x2 - x1) + x1)
    return sdata

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
            ax.set_yticklabels(ax.get_yticklabels(),fontsize=8)   # Increased fontsize for y tick labels
            ax.set_xticklabels(variables,fontsize=6)    # Adjusted fontsize for x tick labels
        
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        #self.ax.set_xticklabels(variables,fontsize=6)
    
    def plot(self,data,*args,**kw):
        sdata=_scale_data(data,self.ranges)      # Ensure _scale_data function works correctly
        self.ax.plot(self.angle,np.r_[sdata,sdata[0]],*args, **kw)        
    
    def fill(self,data,*args,**kw):
        sdata=_scale_data(data,self.ranges)
        self.ax.fill(self.angle,np.r_[sdata, sdata[0]],*args,**kw)
        
# Graph Radar
def radat_id_plot(ID,fig,features=features,fill=False):
    #raw_app_copy = raw_app.copy()    #Create a copy of raw_app
    
    app_id = get_data(raw_app,ID).loc[:,features]
    client = app_id.iloc[0]

    # Modify the copy of raw_app using .loc
    raw_app.loc[raw_app.index[0], 'AGE'] = client['AGE'] - 5
    raw_app.loc[raw_app.index[0], 'YEARS_EMPLOYED'] = client['YEARS_EMPLOYED'] - 1
    raw_app.loc[raw_app.index[0], 'AMT_INCOME_TOTAL'] = client['AMT_INCOME_TOTAL'] - 500
    raw_app.loc[raw_app.index[0], 'AMT_ANNUITY'] = client['AMT_ANNUITY'] - 100
    raw_app.loc[raw_app.index[0], 'AMT_CREDIT'] = client['AMT_CREDIT'] - 500
    
    ranges = [(client['AGE'] -5, client['AGE'] +5),
              (client['YEARS_EMPLOYED'] -1, client['YEARS_EMPLOYED'] +1),
              (client['AMT_INCOME_TOTAL'] -500, client['AMT_INCOME_TOTAL'] +500),
              (client['AMT_ANNUITY'] -100, client['AMT_ANNUITY'] +100),
              (client['AMT_CREDIT']-500, client['AMT_CREDIT'] +500)]
    
    radar = ComplexRadar(fig, features,ranges)
    radar.plot(client,linewidth=3,color='darkseagreen')
    
    if fill:
        radar.fill(client, alpha=0.2)

def radat_knn_plot(ID,fig,fill=False):
    # Get data for the specified client ID
    app_id = get_data(raw_app,ID)[features]
    data_id = app_id.iloc[0]    

    # Get similar IDs using KNN
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna().copy()
    data_knn['TARGET'] = data_knn['TARGET'].astype(int)
    moy_knn = data_knn['TARGET'].groupby('TARGET').mean()
    
    # calculate ranges for radar plot
    ranges = [(min(data_knn['AGE']),max(data_knn['AGE'])),
              (min(data_knn['YEARS_EMPLOYED']),max(data_knn['YEARS_EMPLOYED'])),
              (min(data_knn['AMT_INCOME_TOTAL']),max(data_knn['AMT_INCOME_TOTAL'])),
              (min(data_knn['AMT_ANNUITY']),max(data_knn['AMT_ANNUITY'])),
              (min(data_knn['AMT_CREDIT']),max(data_knn['AMT_CREDIT']))]
    
    # Create radar plot
    radar = ComplexRadar(fig,features,ranges)
    radar.plot(data_id,linewidth=3,label='Client '+str(ID),color='darkseagreen')
    radar.plot(moy_knn.iloc[1][features],linewidth=3,label='Average Similar Client having problems',color='red')
    radar.plot(moy_knn.iloc[0][features],linewidth=3,label='Average similar client without having problems',color='royalblue')
    fig.legend(fontsize=5,loc='upper center',bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    
    if fill:
        radar.fill(client, alpha=0.2)
    
def shap_id(ID,app,X_name,explainer,shap):
    app_id = get_data(app,ID).loc[:, X_name].copy()
    shap_vals = explainer.shap_values(app_id)
    shap.bar_plot(shap_vals[1][0],feature_names=X_name,max_display=10)

# defining Prediction
def predict_target(ID,data,feats,model,st,result):
    #ID=int(ID)
    try:
        ID_data = data.loc[data['SK_ID_CURR'] == ID]
        ID_to_predict = ID_data.loc[feats]        #feature of data_selected1

        # Make predictions
        prediction = model.prediction(ID_to_predict)
        proba = model.predict.proba(ID_to_predict)

        prediction = int(prediction[0])        # Assuming model.predict returns integers

        if prediction in [0, 1]:
            results = {'target': prediction, 'risk':round(proba[0][1],2)}
            st.json(result)
        else:
            st.warning('Error in the program!')
    except:
        st.error('Client not found!')

#if __name__ == '__main__':
    #predict_target(ID)

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
                radat_id_plot(ID,fig,features=features,fill=False)
                #radat_id_plot(ID,fig,features=features,raw_app=raw_app)
                st.pyplot(fig)
                    
        st.markdown("-----")
        
        with st.container():
            st.write("#### Similar type of Customers ")
            #try:
            col3, col4 = st.columns([3,1])
            with col3:
                fig = plt.figure(figsize=(3,3))
                radat_knn_plot(ID,fig)
                st.pyplot(fig)
            with col4:
                N_knn, N_knn1 = get_stat_ID(ID)
                st.markdown("* **Similar type of customers : " + str(N_knn) + "**")
                st.markdown("* **Customer having payment problem : " + str(N_knn1) + "**")                
                st.markdown("_(either " + str(N_knn1*100/N_knn) + "% clients with similar payment problems)_")
                
            
        st.markdown("-----")
        with st.container():
            st.write("#### Customer solvability prediction ")
        prediction_button = st.button('Predict solvability')
                    #pred = st.button('Calculation')
        if prediction_button:
            with st.spinner('Calculating...'):
                try:
                    prediction = predict_target(ID)
                                #prediction = requests.get("https://dashtest.streamlit.app//predict?ID=" + str(ID)).json()
                    if prediction["target"]==0:
                        st.write(':smiley:')
                        st.success('Client solvable _(Target = 0)_, prediction difficulty level at **' + str(prediction["risk"] * 100) + '%**')
                    elif prediction["target"]==1:
                        st.write(':confused:')
                        st.error('Client non solvable _(Target = 1)_, prediction difficult level at **' + str(prediction["risk"] * 100) + '%**')  
                        st.write('**Interpretability**')
                        fig = plt.figure(figsize=(2,2))
                        shap_id(ID)
                        st.pyplot(fig)
                except Exception as e:
                    st.warning('Programme error:'+str(e)) 
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
                fig = plt.figure(figsize=(8,6))
                bins = int((raw_app['AGE'].max() - raw_app['AGE'].min()) // 5)
                bins = max(bins, 1)    #Ensure bins is at least 1                    
                #if bins<0:
                    #bins=5
                    #print('XXXXXXXX',bins)                    
                pt = sns.histplot(data=raw_app, x='AGE', hue='TARGET',bins=bins,palette=['royalblue','red'],alpha=0.5)
                pt.set(xlabel='AGE', ylabel='Frequency')
                pt.legend(['Having difficulty','Without difficulty'],loc='upper right')
                #pt.legend(['Having difficulty', 'Without difficulty'],loc='lower center', bbox_to_anchor=(0.5, -0.35),fancybox=True, shadow=True, ncol=5)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))           # Create a new figure and axis
                # Plot data for TARGET == 1
                sns.barplot(x='NAME_FAMILY_STATUS', y='CNT_CHILDREN', data=raw_app[raw_app['TARGET'] == 1],
                            color='red', alpha=0.5, errorbar=None, edgecolor='black', ax=ax, label='Having difficulty')
                
                            #sns.barplot(x=raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==1],
                            #y=raw_app['CNT_CHILDREN'][raw_app['TARGET']==1],
                            #color='red', alpha=0.5, errorbar=None, edgecolor='black', ax=ax) 

                # Plot data for TARGET == 0
                sns.barplot(x='NAME_FAMILY_STATUS', y='CNT_CHILDREN', data=raw_app[raw_app['TARGET'] == 0],
                            color='royalblue', alpha=0.5, errorbar=None, edgecolor='black', ax=ax, label='Without difficulty')

                            # Plot data for TARGET == 1
                            #sns.barplot(x=raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==0],
                            #y=raw_app['CNT_CHILDREN'][raw_app['TARGET']==0],
                            #color='royalblue', alpha=0.5, errorbar=None, edgecolor='black', ax=ax)
                
                # Customize plot
                ax.set_xlabel('Family Status')
                ax.set_ylabel('Number of Children')
                ax.set_title('Number of Children by Family Status and Target')
                ax.legend()

                 # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                plt.setp(ax.get_yticklabels(), fontsize=8)
    
                #plt.setp(ax.get_xticklabels(), rotation=45, fontsize=7)
                #plt.setp(ax.get_yticklabels(), fontsize=5)

                # Display plot in Streamlit
                st.pyplot(fig)
                
                #fig = plt.figure(figsize=(6,6))
                #pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==1],raw_app['CNT_CHILDREN'][raw_app['TARGET']==1],color='red',alpha=.5,errorbar=None,edgecolor='black')
                #pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==0],raw_app['CNT_CHILDREN'][raw_app['TARGET']==0],color='royalblue',alpha=.5,errorbar=None,edgecolor='black')                    
                #plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                #plt.setp(pt.get_yticklabels(),fontsize=5)
                #st.pyplot(pt.figure)
                
                #subset_data = raw_app[raw_app['TARGET'] == 1]
                #print('Subset data shape:', subset_data.shape)
                #pt = sns.barplot(x=subset_data['NAME_FAMILY_STATUS'], y=subset_data['CNT_CHILDREN'], color='red',alpha=.5,errorbar=None,edgecolor='black')
                #plt.xlabel('Family Status', fontsize=8)
                #plt.ylabel('Number of Children', fontsize=8)
                #pt = sns.barplot(raw_app['NAME_FAMILY_STATUS']==1],raw_app['CNT_CHILDREN'][raw_app['TARGET']==1],color='red',alpha=.5,errorbar=None,edgecolor='black')
                #pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==0],raw_app['CNT_CHILDREN'][raw_app['TARGET']==0],color='royalblue',alpha=.5,errorbar=None,edgecolor='black')
                #plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                #plt.setp(pt.get_yticklabels(),fontsize=5)
                
            with col3:
                # Create a new figure and axis
                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot data for TARGET == 1
                sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=raw_app, hue='TARGET',
                            palette=['royalblue', 'red'], alpha=0.5, edgecolor='black', ax=ax)
                
                #sns.barplot(x=raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==1],
                            #y=raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==1],
                            #color='red', alpha=0.5, edgecolor='black', ax=ax)

                # Plot data for TARGET == 0
                sns.barplot(x=raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==0],
                            y=raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==0],
                            color='royalblue', alpha=0.5, edgecolor='black', ax=ax)

                # Customize plot
                ax.set_xlabel('Income Type')
                ax.set_ylabel('Income Total')
                ax.set_title('Income Total by Income Type and Target')
                ax.legend(title='Target', fontsize=8)

                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                plt.setp(ax.get_yticklabels(), fontsize=8)

                # Display plot in Streamlit
                st.pyplot(fig)
                
                    # Customize plot
                    #plt.setp(ax.get_xticklabels(), rotation=45, fontsize=7)
                    #plt.setp(ax.get_yticklabels(), fontsize=7)

                    #fig = plt.figure(figsize=(6,6))
                    #pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==1],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==1],color='red',alpha=.5,errorbar=None,edgecolor='black')
                    #pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==0],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==0],color='royalblue',alpha=.5,errorbar=None,edgecolor='black')
                    #plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                    #plt.setp(pt.get_yticklabels(),fontsize=7)
                    #st.pyplot(fig)
                
                    #subset_data = raw_app[raw_app['TARGET'] == 1]
                    #pt = sns.barplot(x=subset_data['NAME_INCOME_TYPE'], y=subset_data['AMT_INCOME_TOTAL'], color='red',alpha=.5,errorbar=None,edgecolor='black')
                    #plt.xlabel('Income Type', fontsize=7)
                    #plt.ylabel('Income total', fontsize=7)
                    #st.pyplot(pt.figure)
                    #pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==1],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==1],color='red',alpha=.5,errorbar=None,edgecolor='black')
                    #pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==0],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==0],color='royalblue',alpha=.5,errorbar=None,edgecolor='black')
                    #plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                    #plt.setp(pt.get_yticklabels(),fontsize=7)
                    #st.pyplot(fig)
        
        st.markdown("-----")
               
        with st.container():
            st.write("#### Loan Payment")
            tg_n = np.array([len(raw_app[raw_app['TARGET']==1]),len(raw_app[raw_app['TARGET']==0]),len(raw_app[raw_app['TARGET'].isnull()])])            
            col4, col5 = st.columns(2)
            with col4:
                fig = plt.figure(figsize=(3,3))
                plt.pie(tg_n,labels=['having difficulty','without difficulty','No Loan outstanding'],colors=['red','royalblue','honeydew'],autopct=lambda x:str(round(x,2))+'%')
                st.pyplot(fig)
                    
            # Calculate the number of samples for each target category
            #target_counts = raw_app['TARGET'].value_counts()
            #col4, col5 = st.columns(2)

            # Define labels before calculating target counts
            #labels = ['Having Difficulty', 'Without Difficulty', 'No Loan Outstanding']
           
            # Plot the pie chart if the lengths match
            #if len(target_counts) == len(labels):
                #with col4:
                    #fig, ax = plt.subplots(figsize=(6, 6))
                    #colors = ['red', 'royalblue', 'honeydew']
                    #labels = ['Having Difficulty', 'Without Difficulty', 'No Loan Outstanding']
                    #explode = (0.1, 0, 0)  # explode the first slice (having difficulty)
                    #ax.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%', explode=explode, shadow=True)
                    #ax.set_title('Distribution of Loan Payments')
            
                    # Display the pie chart in Streamlit
                    #st.pyplot(fig)
            #else:
                #st.error("Lengths of target_counts and labels do not match.")
            
                
            
            with col5:
                df = raw_app[['TARGET','NAME_INCOME_TYPE','AMT_ANNUITY','AMT_CREDIT']]
                df.loc[:, 'COUNT_TG'] = df['TARGET']
                        #df.loc[:,'COUNT_TG'] = df['TARGET']
                
                # Group by target and income type and calculate mean annuity and credit amounts, and count of observations
                tg_df = pd.concat((df.groupby(['TARGET','NAME_INCOME_TYPE']).mean()[['AMT_ANNUITY','AMT_CREDIT']],
                                    df.groupby(['TARGET','NAME_INCOME_TYPE']).count()[['COUNT_TG']]), axis = 1)
                tg_0 = tg_df.loc[0]
                tg_1 = tg_df.loc[1]

                # Create scatter plot
                fig = plt.figure(figsize=(3,3))
                sns.scatterplot(x=tg_1['AMT_ANNUITY'], y=tg_1['AMT_CREDIT'], hue=tg_1['COUNT_TG'], palette='coolwarm')
                #sns.scatterplot(tg_0['AMT_ANNUITY'], tg_0['AMT_CREDIT'], s=tg_0['COUNT_TG']/100, label='With difficulty', color='royalblue', alpha=.3)
                #pt = sns.scatterplot(tg_1['AMT_ANNUITY'], tg_1['AMT_CREDIT'], s=tg_1['COUNT_TG'].values/100,label='With difficulty',color='red')
                #pt = sns.scatterplot(tg_0['AMT_ANNUITY'], tg_0['AMT_CREDIT'], s=tg_0['COUNT_TG'].values/100,label='Without difficulty', color='royalblue', alpha=.3)

                #Customize plot
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=5, fontsize=4)
                plt.xlabel('AMT_ANNUITY', fontsize=4)
                plt.ylabel('AMT_CREDIT', fontsize=4)
                plt.xlim([20000, 40000])    ##
                plt.ylim([400000, 800000])  ##
                plt.setp(pt.get_xticklabels(),fontsize=5)
                plt.setp(pt.get_yticklabels(),fontsize=5) 
                
                #Display plot
                st.pyplot(fig)
                
                    #tg_df = df.groupby(['TARGET', 'NAME_INCOME_TYPE']).agg({'AMT_ANNUITY': 'mean', 'AMT_CREDIT': 'mean', 'COUNT_TG': 'count'}).reset_index()
                    #fig, ax = plt.subplots(figsize=(6, 6))               
                    # Create scatter plot with seaborn
                    #fig, ax = plt.subplots(figsize=(8, 6))
                    #sns.scatterplot(data=tg_df, x='AMT_ANNUITY', y='AMT_CREDIT', hue='TARGET', size='COUNT_TG', sizes=(50, 500),
                                    #alpha=0.8, palette={0: 'royalblue', 1: 'red'}, legend='brief', ax=ax)

                     # Customize plot
                    #ax.set_xlabel('Average AMT_ANNUITY', fontsize=12)
                    #ax.set_ylabel('Average AMT_CREDIT', fontsize=12)
                    #ax.set_title('Comparison of Average Annuity and Credit Amounts', fontsize=14)
                    #ax.legend(title='Payment Difficulty', loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=10)
                    #ax.tick_params(axis='both', labelsize=10)            

                    # Display plot
                    #st.pyplot(fig)

                    
