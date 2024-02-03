## Import des librairies
import streamlit as st
import altair as alt              # for data visualtization
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
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
#del raw_train
#del raw_test
 
raw_app['YEARS_EMPLOYED'] = raw_app['DAYS_EMPLOYED'].apply(lambda x : -x/-365)
raw_app['AGE'] = raw_app['DAYS_BIRTH'].apply(lambda x : -x/-365) // (-365)
raw_app['CREDIT'] = raw_app['AMT_CREDIT']   
raw_app['CREDIT'] = raw_app['AMT_CREDIT'].apply(lambda x: 'No' if math.isnan(x) else 'Yes')       

#st.dataframe(raw_app)

# Drop unnecessary columns
raw_app = raw_app.drop(['DAYS_BIRTH','DAYS_EMPLOYED'], axis=1)

# Treated Data

zip_file_path = ZipFile('data_train.zip')
train = pd.read_csv(zip_file_path.open('data_train.csv'))

zip_file_test = ZipFile('data_test.zip')
test = pd.read_csv(zip_file_test.open('data_test.csv'))

#Append/Concat
app = pd.concat([train, test], ignore_index=True)

# Nearest neighbors
knn = NearestNeighbors(n_neighbors=10)
knn.fit(train.drop(['SK_ID_CURR','TARGET'], axis=1))

# Loading the model
model = pickle.load(open('model1.pkl','rb'))

# Explainer
zip_file = ZipFile('X_train_sm_split1.zip')
X_train_sm_1 = pd.read_csv(zip_file.open('X_train_sm_split1.csv'))

zip_file = ZipFile('X_train_sm_split2.zip')
X_train_sm_2 = pd.read_csv(zip_file.open('X_train_sm_split2.csv'))

zip_file = ZipFile('X_train_sm_split3.zip')
X_train_sm_3 = pd.read_csv(zip_file.open('X_train_sm_split3.csv'))

X_train_sm = pd.concat([X_train_sm_1, X_train_sm_2, X_train_sm_3], ignore_index=True)
X_train_sm.reset_index(drop=True, inplace=True)       # Reset the index to have a continuous index for the concatenated DataFrame


#X_train_sm = X_train_sm_split1.append(X_train_sm_split2).reset_index(drop=True).append(X_train_sm_split3).reset_index(drop=True)
#X_name = list(X_train_sm.columns)

#explainer = shap.TreeExplainer(model, X_train_sm)
#shap_values = explainer.shap_values(X_train_sm)

#explainer = shap.LinearExplainer(model, X_train_sm)
#shap_values = explainer.shap_values(X_train_sm)
#explainer = shap.TreeExplainer(model,X_train_sm)

del X_train_sm_1
del X_train_sm_2
del X_train_sm_3
del X_train_sm

# Features
features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']

# Data
def get_data(data,ID):
    if type(ID) == list:
        return data[data['SK_ID_CURR'].isin(ID)]
    else:
        return data[data['SK_ID_CURR']==ID].head(1)

# Neighbor
def get_similar_ID(ID):    
    app_id = app[app['SK_ID_CURR']==ID].drop(['SK_ID_CURR','TARGET'], axis=1)
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
        #inverts a value x on a scale from
        #limits[0] to limits[1]
        return limits[1] - (x - limits[0])

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
        def __init__(self, fig, variables, ranges,
                     n_ordinate_levels=6):
            angles = np.arange(0, 360, 360./len(variables))
    
            axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                    label = "axes{}".format(i)) 
                    for i in range(len(variables))]
            l, text = axes[0].set_thetagrids(angles, 
                                             labels=variables)
            [txt.set_rotation(angle-90) for txt, angle 
                 in zip(text, angles)]
            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)
            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i], 
                                   num=n_ordinate_levels)
                gridlabel = ["{}".format(round(x,2)) 
                             for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1] # hack to invert grid
                              # gridlabels aren't reversed
                gridlabel[0] = "" # clean up origin
                ax.set_rgrids(grid, labels=gridlabel,
                             angle=angles[i])
            ###ax.spines["polar"].set_visible(False)
                ax.set_ylim(*ranges[i])
                ax.set_yticklabels(ax.get_yticklabels(),fontsize=4)                       
        
            # variables for plotting
            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]
            self.ax.set_xticklabels(variables,fontsize=6) 
        def plot(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)        
        def fill(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        
    # Graph Radar
    def radat_id_plot(ID,fig,features=features,fill=False):
        app_id = get_data(raw_app,ID)[features]
        client = app_id.iloc[0]
        ranges = [(client['AGE']-5, client['AGE']+5),
                  (client['YEARS_EMPLOYED']-1, client['YEARS_EMPLOYED']+1),
                  (client['AMT_INCOME_TOTAL']-500, client['AMT_INCOME_TOTAL']+500),
                  (client['AMT_ANNUITY']-100, client['AMT_ANNUITY']+100),
                  (client['AMT_CREDIT']-500, client['AMT_CREDIT']+500)]
    
        radar = ComplexRadar(fig,features,ranges)
        radar.plot(client,linewidth=3,color='darkseagreen')
        if fill:
            radar.fill(client, alpha=0.2)
        
    def radat_knn_plot(ID,fig,features=features,fill=False):
        app_id = get_data(raw_app,ID)[features]
        data_id = app_id.iloc[0]    
        app_knn = get_similar_ID(ID)
        data_knn = get_data(raw_app,app_knn).dropna()
        data_knn['TARGET'] = data_knn['TARGET'].astype(int)
        moy_knn = data_knn.groupby('TARGET').mean()
        ranges = [(min(data_knn['AGE']), max(data_knn['AGE'])),
                  (min(data_knn['YEARS_EMPLOYED']),  max(data_knn['YEARS_EMPLOYED'])),
                  (min(data_knn['AMT_INCOME_TOTAL']),  max(data_knn['AMT_INCOME_TOTAL'])),
                  (min(data_knn['AMT_ANNUITY']),  max(data_knn['AMT_ANNUITY'])),
                  (min(data_knn['AMT_CREDIT']),  max(data_knn['AMT_CREDIT']))]
        
        radar = ComplexRadar(fig,features,ranges)
        radar.plot(data_id,linewidth=3,label='Client '+str(ID),color='darkseagreen')
        radar.plot(moy_knn.iloc[1][features],linewidth=3,label='Client similaire moyen avec difficultés',color='red')
        radar.plot(moy_knn.iloc[0][features],linewidth=3,label='Client similaire moyen sans difficultés',color='royalblue')
        fig.legend(fontsize=5,loc='upper center',bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
        if fill:
            radar.fill(client, alpha=0.2)
    
    def shap_id(ID):
        app_id = get_data(app,ID)[X_name]
        shap_vals = explainer.shap_values(app_id)
        shap.bar_plot(shap_vals[1][0],feature_names=X_name,max_display=10)


###############################################
## DASH BOARD
########################

    ## Page configuration initialisation
st.set_page_config(
page_title="Credit Score Dashboard",
page_icon="💵",
layout="wide",
initial_sidebar_state="expanded")
    
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
    
st.write("By: Amit GAUTAM")
    


if page == "Home":
    st.title("💵 Credit Score Dashboard - Customer Page")
    ".\n"
    #.\n"
        
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                    
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer**: to find out all the information related to the customer.\n")
                    
if page == "Customer":
    st.title("💵 Welcome to the Customer Page")
    ".\n"
                
    #st.header("Welcome to the customers' page.\n")
    ".\n"
    st.subheader("Please enter your ID to know the results of your demands. \n") 
    #"Thank you. \n"
    st.markdown("Your ID:")
    ID=st.number_input(" ", min_value=100002, max_value=456255)
    try:
        raw_app_id = get_data(raw_app, ID)
        with st.spinner('Custumer....'):
            st.writer('Customer .....')
            with st.container():
                col1, col2 = st.columns([1.5,2.5])      
                with col1:
                    st.write("#### Customer detail " + str(ID))
                    st.markdown("* **Status : " + str(id_raw_app['NAME_FAMILY_STATUS'].values[0]) + "**")
                    st.markdown("* **Number of children) : " + str(id_raw_app['CNT_CHILDREN'].values[0]) + "**")
                    st.markdown("* **Employment: " + str(id_raw_app['NAME_INCOME_TYPE'].values[0]) + "**")
                    st.markdown("* **Current Loan : " + str(id_raw_app['CREDIT'].values[0]) + "**")
                with col2:
                    fig = plt.figure(figsize=(2,2))
                    st.pyplot(radat_id_plot(ID,fig))
            st.markdown("-----")
    
            with st.container():
                st.write("#### Similar type of Customers ")
                try:
                    col3, col4 = st.columns([3,1])
                    with col3:
                        fig = plt.figure(figsize=(3,3))
                        st.pyplot(radat_knn_plot(ID,fig))
                    with col4:
                        N_knn, N_knn1 = get_stat_ID(ID)
                        st.markdown("* **Similar type of customers : " + str(N_knn) + "**")
                        st.markdown("* **Customer having payment problem : " + str(N_knn1) + "**")                
                        st.markdown("_(either " + str(N_knn1*100/N_knn) + "% clients with similar payment problems)_")
                except:
                    st.info('**_No similar customer_**')
            
            st.markdown("-----")
            with st.container():
                st.write("#### Customer solvability prediction ")
                pred = st.button('Calculation')
                if pred:
                    with st.spinner('Calculation...'):
                        try:
                            prediction = requests.get("https://urd9pbjwdlnjfnaoncmtdw.streamlit.app/predict?ID=" + str(ID)).json()
                            if prediction["target"]==0:
                                st.write(':smiley:')
                                st.success('Client solvable _(Target = 0)_, prediction difficult level at **' + str(prediction["risk"] * 100) + '%**')
                            elif prediction["target"]==1:
                                st.write(':confused:')
                                st.error('Client non solvable _(Target = 1)_, prediction difficult level at **' + str(prediction["risk"] * 100) + '%**')  
                            st.write('**Interpretability**')
                            fig = plt.figure(figsize=(2,2))
                            st.pyplot(shap_id(ID))
                        except :
                            st.warning('programme error programme') 
                            st.write(':dizzy_face:')                                               
                    
    except:
        st.warning('**_Customer not found_**')

# Customer portfolio analysis        
if page == 'Customer portfolio':
    st.write("### Customer portfolio analysis")
    with st.spinner('Analysing...'):
        with st.container():            
            st.write("#### Customer Profile")
            col1, col2,col3 = st.columns(3)
            plt.ioff()
            with col1:
                fig = plt.figure(figsize=(4,4))
                bins = int(raw_app['AGE'].max()-raw_app['AGE'].min()//5)
                print('XXXXXXXX',bins)                    
                pt = sns.histplot(data=raw_app, x='AGE', hue='TARGET',bins=bins,palette=['royalblue','red'],alpha=.5)
                plt.xlabel('AGE',fontsize=12)
                plt.ylabel('')
                plt.legend(['having difficulty','without difficulty'],loc='lower center',bbox_to_anchor=(0.5, -0.35),fancybox=True, shadow=True, ncol=5)
                st.pyplot(fig)
            with col2:
                fig = plt.figure(figsize=(3,3))                
                pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==1],raw_app['CNT_CHILDREN'][raw_app['TARGET']==1],color='red',alpha=.5,ci=None,edgecolor='black')
                pt = sns.barplot(raw_app['NAME_FAMILY_STATUS'][raw_app['TARGET']==0],raw_app['CNT_CHILDREN'][raw_app['TARGET']==0],color='royalblue',alpha=.5,ci=None,edgecolor='black')
                plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                plt.setp(pt.get_yticklabels(),fontsize=5)
                st.pyplot(fig)
            with col3:
                fig = plt.figure(figsize=(4.5,4.5))
                pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==1],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==1],color='red',alpha=.5,ci=None,edgecolor='black')
                pt = sns.barplot(raw_app['NAME_INCOME_TYPE'][raw_app['TARGET']==0],raw_app['AMT_INCOME_TOTAL'][raw_app['TARGET']==0],color='royalblue',alpha=.5,ci=None,edgecolor='black')
                plt.setp(pt.get_xticklabels(),rotation=45,fontsize=7)
                plt.setp(pt.get_yticklabels(),fontsize=7)
                st.pyplot(fig)
            st.markdown("-----")

                
        with st.container():
            st.write("#### Loan Payment")
            tg_n = np.array([len(raw_app[raw_app['TARGET']==1]),len(raw_app[raw_app['TARGET']==0]),len(raw_app[raw_app['TARGET'].isnull()])])            
            col4, col5 = st.columns(2)
            with col4:
                fig = plt.figure(figsize=(5,5))
                plt.pie(tg_n,labels=['having difficulty','without difficulty','No Loan outstanding'],colors=['red','royalblue','honeydew'],autopct=lambda x:str(round(x,2))+'%')
                st.pyplot(fig)
            with col5:
                df = raw_app[['TARGET','NAME_INCOME_TYPE','AMT_ANNUITY','AMT_CREDIT']]
                df['COUNT_TG'] = df['TARGET']
                tg_df = pd.concat((df.groupby(['TARGET','NAME_INCOME_TYPE']).mean()[['AMT_ANNUITY','AMT_CREDIT']],df.groupby(['TARGET','NAME_INCOME_TYPE']).count()[['COUNT_TG']]), axis = 1)
                tg_0 = tg_df.loc[0]
                tg_1 = tg_df.loc[1]
                fig = plt.figure(figsize=(2,2))                  
                pt = sns.scatterplot(tg_1['AMT_ANNUITY'],tg_1['AMT_CREDIT'],s=tg_1['COUNT_TG'].values/100,label='Avec Difficulté',color='red')
                pt = sns.scatterplot(tg_0['AMT_ANNUITY'],tg_0['AMT_CREDIT'],s=tg_0['COUNT_TG'].values/100,label='Sans Difficulté',color='royalblue',alpha=.3)
                plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True, ncol=5,fontsize=5)
                plt.xlabel('AMT_ANNUITY',fontsize=5)
                plt.ylabel('AMT_CREDIT',fontsize=5)
                plt.xlim([20000,40000])
                plt.ylim([400000,800000])
                plt.setp(pt.get_xticklabels(),fontsize=4)
                plt.setp(pt.get_yticklabels(),fontsize=4)                
                st.pyplot(fig)
