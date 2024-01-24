## Import des librairies
import shap
import streamlit as st
import altair as alt              # for data visualtization
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests
import plotly
import os
from zipfile import ZipFile
import pickle
import math


# Features
feat = ['SK_ID_CURR','TARGET','DAYS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
        'DAYS_EMPLOYED','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']

# Original Data
raw_train = pd.read_csv(('application_train.csv'),usecols=feat)
#zip_file = ZipFile('data/application_train.zip')
#raw_train = pd.read_csv(zip_file.open('application_train.csv'),usecols=feat)

#zip_file = ZipFile('data/application_test.zip')
#raw_test = pd.read_csv(zip_file.open('application_test.csv'),usecols=[f for f in feat if f!='TARGET'])

#zip_file_train= './data/application_train.zip'
#with ZipFile(zip_file_train, 'r') as zip_app_train:
    #raw_train = pd.read_csv(zip_app_train.open('application_train.csv'),usecols=feat)
#zip_file_test= './data/application_test.zip'
#with ZipFile(zip_file_test, 'r') as zip_app_test:
    #raw_test = pd.read_csv(zip_app_test.open('application_test.csv'),usecols=[f for f in feat if f!='TARGET'])
raw_app = raw_train.append(raw_test).reset_index(drop=True)
del raw_train
del raw_test
raw_app['AGE'] = raw_app['DAYS_BIRTH'] // (-365)
raw_app['YEARS_EMPLOYED'] = raw_app['DAYS_EMPLOYED'] // (-365)
raw_app['CREDIT'] = raw_app['AMT_CREDIT'].apply(lambda x: 'No' if math.isnan(x) else 'Yes')
# Drop unnecessary columns
raw_app = raw_app.drop(['DAYS_BIRTH','DAYS_EMPLOYED'], axis=1)
# Treated Data
zip_file_train = './data_train.zip'
zip_file_test = './data_test.zip'
with ZipFile(zip_file_train, 'r') as zip_train:
    train = pd.read_csv(zip_train.open('data_train.csv'))

with ZipFile(zip_file_test, 'r') as zip_test:
    test=pd.read_csv(zip_test.open('data_test.csv'))    

app = train.append(test).reset_index(drop=True)

# Modele voisin
knn = NearestNeighbors(n_neighbors=10)
knn.fit(train.drop(['SK_ID_CURR','TARGET'], axis=1))

# Chargement du modèle de classification
pk_mdl_in = open('Results/model.pkl','rb')
model = pickle.load(pk_mdl_in)


# Explainer
#zip_file = ZipFile('Results/X_train_sm_0.zip')
#X_train_sm_0 = pd.read_csv(zip_file.open('X_train_sm_0.csv'))
#zip_file = ZipFile('Results/X_train_sm_1.zip')
#X_train_sm_1 = pd.read_csv(zip_file.open('X_train_sm_1.csv'))
#zip_file = ZipFile('Results/X_train_sm_2.zip')
#X_train_sm_2 = pd.read_csv(zip_file.open('X_train_sm_2.csv'))
#X_train_sm = X_train_sm_0.append(X_train_sm_1).reset_index(drop=True).append(X_train_sm_2).reset_index(drop=True)

X_name = list(X_train_sm.columns)

explainer = shap.TreeExplainer(model,X_train_sm)
    
# Features
features =['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT']

# Recuperation de data
def get_data(data,ID):
    if type(ID) == list:
        return data[data['SK_ID_CURR'].isin(ID)]
    else:
        return data[data['SK_ID_CURR']==ID].head(1)

# Recuperation des voisins
def get_similar_ID(ID):    
    app_id = app[app['SK_ID_CURR']==ID].drop(['SK_ID_CURR','TARGET'], axis=1)
    knn_index = knn.kneighbors(app_id,return_distance=False)
    knn_id = app['SK_ID_CURR'][app.index.isin(knn_index[0])].values.tolist()
    return knn_id

def get_stat_ID(ID):   
    app_knn = get_similar_ID(ID)
    data_knn = get_data(raw_app,app_knn).dropna()
    return len(data_knn),len(data_knn[data_knn['TARGET']==1])

## GRAPHE
# Initialisation de Graphe Radar
def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
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
            #ax.spines["polar"].set_visible(False)
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
        
# Graphe Radar
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
    #shap.force_plot(explainer.expected_value[1], shap_vals[1], app_id)























########################
# DASHBOARD

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
    page =  st.selectbox("Menu", ["Home", "Customer"])


    
    st.markdown("""---""")
    
    st.write("By: Amit GAUTAM")



if page == "Home":
    st.title("💵 Credit Score Dashboard - Home Page")
    ".\n"
    #.\n"
    
    st.markdown("This is an interactive dashboard website which lets the clients to know about their credit demands\n"
                "approved ou refused. The predictions are calculted automatically with the help of machine learning algorithm.\n"
                                
                "\nThis dashboard is composed of following pages :\n"
                "- **Customer**: to find out all the information related to the customer.\n")
                
if page == "Customer":
    st.title("💵 Welcome to the Customer Page")
    ".\n"
    #.\n"
    
    st.header("Welcome to the customers' page.\n")
    ".\n"
    st.subheader("Please enter your ID to know the results of your demands. \n") 
    #"Thank you. \n"
    st.markdown("Your ID:")
    Cst_ID=st.number_input(" ", min_value=100002, max_value=456255)

     
    #with st.sidebar:
        #st.selectbox("Enter your ID", "      ")
        #button_start = st.button("Submit")
    
    
    
                #if button_start:
# Use the entered customer_id to fetch and display relevant information
# Add your logic here
                #customer_info = fetch_customer_info("Navigation")
                #st.write(customer_info)

    #Id selection
 
# First, create the layout with three columns
    col1, col2, col3 = st.columns(3)

# Now, add content to each column
    with col1:
        st.header("Column 1")
        st.write("Content for column 1 goes here.")

    with col2:
        st.header("Column 2")
        st.write("Content for column 2 goes here.")

    with col3:
        st.header("Column 3")
        st.write("Content for column 3 goes here.")
