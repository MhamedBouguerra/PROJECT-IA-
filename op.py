from select import select
from ssl import Options
import streamlit as st
from PIL import Image
import base64
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import pickle
#----importation Matricule------
from matplotlib import pyplot as plt
import imutils
#import easyocr
import random
import cv2
import numpy as np
import plotly.express as px
#Parti Analyse sentimental
from streamlit_lottie import st_lottie
import json
import requests
#machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import datetime


#page icon & titre
st.set_page_config(page_title="IA PROJET MHAMED", page_icon="favicon.jpg", layout="wide")


#login
#st.title("information d'application")
st.header("Introduction de projet")
st.markdown("WEREACT PROJET IA.")
#st.markdown("Ce site recueille toutes les fonctionnalités d'un data scientist en un seul projet, nous avons utilisé l'apprentissage automatique, l'intelligence artificielle, l'analyse du langage humain et l'analyse des données, Afin d'obtenir un programme complet pour un data scientist.")

st.header("")
st.header("")
st.header("")



i7,i8,i9=st.columns(3)
with i7:
    i77 = Image.open('lml.png')
    st.image(i77, width=330)

with i8:
    i88 = Image.open('IA.png')
    st.image(i88, width=250)

with i9:
    i99 = Image.open('bi.png')
    st.image(i99, width=330)


image = Image.open('wereact2.png')
#st.sidebar.image(image, width=100)
with open('wereact2.png', 'rb') as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
st.sidebar.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{image_base64}" width="150"></div>',
    unsafe_allow_html=True
)
username=st.sidebar.text_input("Nom d'utilisateur")
password= st.sidebar.text_input("Mot de passe",type='password')
if st.sidebar.checkbox("Connexion"):
    if username=="mhamed" and password=="bou123":
        st.success("Connecté en tant que {}".format(username))

#Head
        selected = option_menu(
            menu_title=None,
            options=['PREDICTION DE FIABILITÉ','CLASSIFICATION DES CATEGORIES','DASHBORED BI',"ANALYSE"],
            icons = ['tv','person','heart'],
            menu_icon="cast",
            default_index = 0,
            orientation="horizontal",)


        #database 
        data=pd.read_csv('algo.csv')
        data=data.drop(['Unnamed: 0'],axis=1)
        dataml=pd.read_csv('ML.csv')
        dataml=dataml.drop(['Unnamed: 0'],axis=1)


        #Parti Detection de fraude
        if (selected == 'PREDICTION DE FIABILITÉ'):
            #page title 
             #traitement
            X=dataml.drop(columns=["identificationTiers","libUsage","jour_dateEcheancePolice","mois_dateEcheancePolice","VolontaireAccidents","CountSinisterAdvr","typeIntermediaire","vehicule_id","assure_id","CONTRAT_EN_COURS","pourcentadeDeResponsabilite","porcentageCompagnieAdverse","Duree","Fraud"])
            Y=dataml['Fraud']
            X_train, X_test, y_train, y_test = train_test_split(X.values,Y.values, test_size = 0.2, random_state = 1)
            xgb_clfdf = XGBClassifier()
            xgb_clfdf.fit(X_train, y_train)

            #colomun for input filed
            col1,col2,col3=st.columns(3)
            with col1:
                codecmp= st.text_input('Saisir code de compagnie')
            with col2:
                nbsi= st.text_input('Saisir nombre de sinistre')
            with col3:
                naturepolice=st.selectbox('Sélect nature de police',('Sélect...','T','R'))
            with col1:
                etatpolice= st.selectbox('Sélect état police',('Sélect...','V','R','Autre'))
            with col2:
                clssbnsmalus= st.text_input('Saisir Classe bonus-malus')
            with col3:
                puissancefiscall= st.selectbox('Sélect puissance fiscal',('Sélect...','Essance','Gasoil','ESS-GAZ GPL','Autre'))
            with col1:
                energieee= st.text_input('Saisir énergie')
            with col2:
                etatvehiculee= st.selectbox('Sélect état véhicule',('Sélect...','V','E'))
            with col3:
                naturesinistre= st.selectbox('Sélect nautre du sinistre',('Sélect...','M','C'))

            with col1:
                dateeffet= st.date_input("Saisir date effet police",datetime.date(2020,8,1))
                dayef = int(dateeffet.strftime("%d"))
                moisef = int(dateeffet.strftime("%m"))
                anneef = int(dateeffet.strftime("%Y"))
            with col2:
                datecalcul=st.date_input("Saisir date de calcule",datetime.date(2020,8,1))
                daycal = int(datecalcul.strftime("%d"))
                moiscal= int(datecalcul.strftime("%m"))
                annecal = int(datecalcul.strftime("%Y"))
            with col3:
                ovrtsinistra=st.date_input("Saisir date ouverture de sinistre",datetime.date(2018,8,1))
                dayo = int(ovrtsinistra.strftime("%d"))
                moiso = int(ovrtsinistra.strftime("%m"))
                anneo = int(ovrtsinistra.strftime("%Y"))

            #creation de button prediction
            if st.button('Résultat'):
                #conduction de test
                if naturepolice=="T":
                    npolice=1
                else:
                    npolice=0

                if etatpolice=="V":
                    netatp=2
                elif etatpolice=="R":
                    netatp=0
                else:
                    netatp=1

                if puissancefiscall=="Essance":
                    puisfis=2
                elif puissancefiscall=="Gasoil":
                    puisfis=3
                elif puissancefiscall=="ESS-GAZ GPL":
                    puisfis=1
                else:
                    puisfis=0

                if etatvehiculee=="V":
                    etatvh=1
                else:
                    etatvh=0
                
                if naturesinistre=="M":
                    natsin=1
                else:
                    natsin=0
                
                codecmp=int(codecmp)
                nbsi=int(nbsi)
                clssbnsmalus=int(clssbnsmalus)
                energieee=int(energieee)


                #prédiction 
                y_pred_XGB = xgb_clfdf.predict([[codecmp,nbsi,npolice,netatp,clssbnsmalus,puisfis,energieee,etatvh,natsin,
                anneo,anneef,moisef,dayef,annecal,moiscal,daycal,moiso,dayo]])

                py_pred_XGB=int(y_pred_XGB)
                if y_pred_XGB==1:
                    st.error("Ce client est fraude 🚨🚨")
                else:
                    st.success("Ce client n'est pas fraude ✅✅")


        #Parti Detection Matricule
        if (selected == 'CLASSIFICATION DES CATEGORIES'):
            st.title('CLASSIFICATION DES CATEGORIES')
            
            # Add a file uploader for the dataset
            dataset = st.file_uploader("Téléchargez votre dataset ici", type=["csv"])
            
                
            # Display the dataset and add a button to download it
            if dataset is not None:
                df = pd.read_csv(dataset)
                st.dataframe(df)
                
                
                # Add a selectbox to choose a column
                column = st.selectbox('Choisissez colonne Ciblé :', df.columns)
                

                
                # Add a button to create a new dataset with just the selected column
                if st.button('Prediction'):
                    new_df = df[[column]]

                    #Pretraitement 
                    import re
                    import nltk
                    from nltk.corpus import stopwords
                    from nltk.stem import WordNetLemmatizer

                    def cleantwt(twt):
                        emoj = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                                        "]+", re.UNICODE)
                        twt = re.sub('RT', '', twt) # remove 'RT' from tweets
                        twt = re.sub('#[A-Za-z0-9]+', '', twt) # remove the '#' from the tweets
                        twt = re.sub('\\n', '', twt) # remove the '\n' character
                        twt = re.sub('https?:\/\/\S+', '', twt) # remove the hyperlinks
                        twt = re.sub('@[\S]*', '', twt) # remove @mentions
                        twt = re.sub('^[\s]+|[\s]+$', '', twt) # remove leading and trailing whitespaces
                        twt = re.sub(emoj, '', twt) # remove emojis
                        return twt

                    # Fonction de prétraitement de texte
                    def preprocess_text(text):
                        # Tokenisation
                        tokens = nltk.word_tokenize(text.lower())
                        # Suppression des stopwords
                        stop_words = set(stopwords.words('french'))
                        filtered_tokens = [token for token in tokens if token not in stop_words]
                        # Lemmatisation
                        lemmatizer = WordNetLemmatizer()
                        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
                        # Retourne une chaîne de caractères des tokens prétraités
                        return " ".join(lemmatized_tokens)
                    
                    df[column] = df[column].apply(cleantwt)   #prétraitement de texte
                    df[column] = df[column].apply(lambda x: preprocess_text(str(x)) if not pd.isnull(x) else np.nan)

                    #import the model
                    import pickle
                    with open('mhamedmodelto.pkl', 'rb') as f:
                        clf3 = pickle.load(f)

                    #train the model
                    predicted = clf3.predict(df[column])
                    df["Predicted"] = predicted

                    #Categorisation avec le code 
                    df.loc[df['Predicted'] == 0, 'Predictedcat'] = 'Accessoires'
                    df.loc[df['Predicted'] == 2, 'Predictedcat'] = 'Chaussures'
                    df.loc[df['Predicted'] == 3, 'Predictedcat'] = 'Cuisine & Électroménager'
                    df.loc[df['Predicted'] == 4, 'Predictedcat'] = 'Informatique'
                    df.loc[df['Predicted'] == 5, 'Predictedcat'] = 'Jardin & Plein air'
                    df.loc[df['Predicted'] == 6, 'Predictedcat'] = 'Maison & Bureau'
                    df.loc[df['Predicted'] == 7, 'Predictedcat'] = 'Mode'
                    df.loc[df['Predicted'] == 8, 'Predictedcat'] = 'Santé & Beauté'
                    df.loc[df['Predicted'] == 9, 'Predictedcat'] = 'Superette'
                    df.loc[df['Predicted'] == 10, 'Predictedcat'] = 'Téléphone & Tablette'
                    df.loc[df['Predicted'] == 11, 'Predictedcat'] = 'autres'
                    df.loc[df['Predicted'] == 12, 'Predictedcat'] = 'Électroniques'

                    #affichage de resultat dataset
                    st.dataframe(df)

                    # Add a button to download the dataset
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Télécharger le dataset complet</a>'
                    st.markdown(href, unsafe_allow_html=True)


        #parti Analyse sentimental
        if (selected == "DASHBORED BI"):

            # Embed Power BI report
            iframe_code = '<iframe title="GLOBAL" width="1170" height="650" src="https://app.powerbi.com/reportEmbed?reportId=85e01ac0-b295-4985-9e6f-27b6ac30a481&autoAuth=true&ctid=dbd6664d-4eb9-46eb-99d8-5c43ba153c61&navContentPaneEnabled=false&chromeless=true" frameborder="0" allowFullScreen="true"></iframe>'
            st.write(iframe_code, unsafe_allow_html=True)


            #if i want to in new dataset or modif
            df=pd.read_csv('rslt.csv')
            df=df.drop(['Unnamed: 0'],axis=1)
            df["sentiment"].replace({'Neutral':'Neutre'}, inplace=True)
            df.rename(columns={'Month': 'Mois'}, inplace=True)
            df.rename(columns={'Year': 'ANNÉE'}, inplace=True)
            df.rename(columns={'Comment': 'Commentaires'}, inplace=True)


        if (selected == "ANALYSE"):
            #dataset
            dff=pd.read_csv('algo.csv')
            dff=dff.drop(['Unnamed: 0'],axis=1)

            #tbadél les cors el liste
            fraudd= dff['Fraud'].unique().tolist()
            codeCompagniee=dff['codeCompagnie'].unique().tolist()
            naturePolicee=dff['natureDuSinistre'].unique().tolist()

            #parti ajnab
            #sidebar
            st.sidebar.header("Veuillez filtrer ici :")

            fraudd=st.sidebar.multiselect(
                "Sélectionner le type de fraude :",
                options=fraudd,
                default=fraudd)

            codeCompagniee=st.sidebar.multiselect(
                "Sélectionner le code de la compagnie :",
                options=codeCompagniee,
                default=codeCompagniee)

            naturePolicee=st.sidebar.multiselect(
                "Sélectionner la nature de sinistre:",
                options=naturePolicee,
                default=naturePolicee)

            st.sidebar.write('---------------------------')
            st.sidebar.write("Fraude 0 : client n'est pas fraude")
            st.sidebar.write('Fraude 1 : client fraude')
            st.sidebar.write('---------------------------')
            st.sidebar.write("Code compagnie 1 : code de l'agence STAR")
            st.sidebar.write("Code compagnie 2 : code de l'agence  COMAR")
            st.sidebar.write("Code compagnie 4 : code de l'agence  DAR ETTAAMIN")
            st.sidebar.write("Code compagnie 6 : code de l'agence  MAE")
            st.sidebar.write("Code compagnie 9 : code de l'agence  GAT")
            st.sidebar.write('---------------------------')
            st.sidebar.write("Nature de sinistre (M) : un accident MATÉRIEL")
            st.sidebar.write("Nature de sinistre (C): un accident CORPOREL")
            st.sidebar.write('---------------------------')

            #end parti 
            blank,tit,blankk= st.columns(3)
            tit.title(":tractor: ANALYSE  :car:")
            st.markdown("##")


            #condution de dataset
            maskk = (dff['natureDuSinistre'].isin(naturePolicee)) & (dff['Fraud'].isin(fraudd)) & (dff['codeCompagnie'].isin(codeCompagniee))

            #créat le number
            totall = int(dff[maskk].shape[0])
            code=dff[maskk].codeCompagnie.unique()
            mo=str(dff[maskk].natureDuSinistre.unique())
            stars= ":speech_balloon:"
            datee= ":construction:"
            office= ":office:"

            col1,col2,col3=st.columns(3)
            with col1:
                st.markdown("CODE COMPAGNIE")
                st.subheader(f"{code:} {office}")

            with col2:
                st.markdown("NOMBRES DES TRANSACTIONS")
                st.subheader(f"{totall:} {stars}")

            with col3:
                st.markdown("NATURE DU SINISTRE")
                st.subheader(f"{mo:} {datee}")

            st.markdown("---")


            #Graphes--
            col77,col88=st.columns(2)
            with col77:
                df_grouped1 = dff[maskk].groupby(by=['Fraud']).count()[['vehicule_id']]
                df_grouped1 = df_grouped1.reset_index()
                bar_chart11 = px.bar(df_grouped1,
                            x='Fraud',
                            y='vehicule_id',
                            text='vehicule_id',
                            color_discrete_sequence = ['#46b7e3']*len(df_grouped1),
                            template= 'plotly_white')
                st.plotly_chart(bar_chart11)

                with col88:
                    df_grouped2 = dff[maskk].groupby(by=['codeCompagnie']).count()[['Fraud']]
                    df_grouped2 = df_grouped2.reset_index()
                    bar_chart22 = px.bar(df_grouped2,
                        x='codeCompagnie',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#b03c28']*len(df_grouped2),
                        template= 'plotly_white')
                    st.plotly_chart(bar_chart22)

            st.dataframe(dff)



            col100,col101=st.columns(2)
            with col100:
                df_grouped3 = dff[maskk].groupby(by=['classeBonusMalus']).count()[['Fraud']]
                df_grouped3 = df_grouped3.reset_index()
                bar_chart33 = px.bar(df_grouped3,
                    x='classeBonusMalus',
                    y='Fraud',
                    text='Fraud',
                    color_discrete_sequence = ['#c7821c']*len(df_grouped3),
                    template= 'plotly_white')
                st.plotly_chart(bar_chart33)

            with col101:
                pie_chart100 = px.pie(dff[maskk],title="",values="CONTRAT_EN_COURS",labels = ['Contrat en cours', 'Contrat Résilié'],names="Fraud")
                plt.legend()
                st.plotly_chart(pie_chart100)


            col102,col103=st.columns(2)
            with col102:
                df_grouped4 = dff[maskk].groupby(by=['energie']).count()[['Fraud']]
                df_grouped4 = df_grouped4.reset_index()
                bar_chart44 = px.bar(df_grouped4,
                    x='energie',
                    y='Fraud',
                    text='Fraud',
                    color_discrete_sequence = ['#a9ba0d']*len(df_grouped4),
                    template= 'plotly_white')
                st.plotly_chart(bar_chart44)

            with col103:
                df_grouped5 = dff[maskk][~(dff[maskk].puissanceFiscal >13 )].groupby(by=['puissanceFiscal']).count()[['Fraud']]
                df_grouped5 = df_grouped5.reset_index()
                bar_chart55 = px.bar(df_grouped5,
                    x='puissanceFiscal',
                    y='Fraud',
                    text='Fraud',
                    color_discrete_sequence = ['#616b06']*len(df_grouped5),
                    template= 'plotly_white')
                st.plotly_chart(bar_chart55)


            col104,col105=st.columns(2)
            with col104:
                df_grouped6 = dff[maskk].groupby(by=['libUsage']).count()[['Fraud']]
                df_grouped6 = df_grouped6.reset_index()
                bar_chart66 = px.bar(df_grouped6,
                    x='libUsage',
                    y='Fraud',
                    text='Fraud',
                    color_discrete_sequence = ['#6b0615']*len(df_grouped6),
                    template= 'plotly_white')
                st.plotly_chart(bar_chart66)

            with col105:
                def load_lottiefile(filepath: str):
                    with open(filepath, "r") as f:
                        return json.load(f)
                
                def load_lottieurl(url: str):
                    r = requests.get(url)
                    if r.status_code != 200:
                        return None
                    return r.json()

                lottie_codingusa= load_lottiefile("coding3.json")
                lottie_usa=load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_vhgwa3yi.json")
                st_lottie(
                    lottie_codingusa,
                    speed=1,
                    reverse=False,
                    loop=False,
                    quality="low",
                    height=0.5,
                    width=0.1,
                    key=None,
                )
                st_lottie(lottie_usa,key="typecar")

            dataMMM = pd.DataFrame({'date': ["2017-05-08","2017-04-17","2017-04-11","2017-02-20","2017-03-06","2017-04-24","2017-04-21","2017-05-02","2017-03-08","2017-03-07","2017-03-30","2017-04-03"], 'Count': [126,119,113,107,104,104,104,104,101,100,99,97]})
            dataKKKK=[]
            for i in range(dataMMM.Count[0]):
                    dataKKKK.append(dataMMM.date[0])
            for i in range(dataMMM.Count[1]):
                    dataKKKK.append(dataMMM.date[1])
            for i in range(dataMMM.Count[2]):
                    dataKKKK.append(dataMMM.date[2])
            for i in range(dataMMM.Count[3]):
                    dataKKKK.append(dataMMM.date[3])
            for i in range(dataMMM.Count[4]):
                    dataKKKK.append(dataMMM.date[4])
            for i in range(dataMMM.Count[5]):
                    dataKKKK.append(dataMMM.date[5])
            for i in range(dataMMM.Count[6]):
                    dataKKKK.append(dataMMM.date[6])
            for i in range(dataMMM.Count[7]):
                    dataKKKK.append(dataMMM.date[7])
            for i in range(dataMMM.Count[8]):
                    dataKKKK.append(dataMMM.date[8])
            for i in range(dataMMM.Count[9]):
                    dataKKKK.append(dataMMM.date[9])
            for i in range(dataMMM.Count[10]):
                    dataKKKK.append(dataMMM.date[10])
            for i in range(dataMMM.Count[11]):
                    dataKKKK.append(dataMMM.date[11])
            dataVVVV = pd.DataFrame()
            dataVVVV['date']=dataKKKK

            col108,col109=st.columns(2)
            with col108:
                dataR = pd.DataFrame({'DATE_RESILIATION': ["2018-01-01","2018-02-02","2017-11-30","2017-12-07","2017-10-10","2017-12-26","2018-01-10","2018-01-09","2018-01-05","2017-11-20","2017-12-08","2018-02-07"], 'Count': [76,68,30,28,26,26,26,25,25,25,25,23]})
                dataKK=[]
                for i in range(dataR.Count[0]):
                        dataKK.append(dataR.DATE_RESILIATION[0])
                for i in range(dataR.Count[1]):
                        dataKK.append(dataR.DATE_RESILIATION[1])
                for i in range(dataR.Count[2]):
                        dataKK.append(dataR.DATE_RESILIATION[2])
                for i in range(dataR.Count[3]):
                        dataKK.append(dataR.DATE_RESILIATION[3])
                for i in range(dataR.Count[4]):
                        dataKK.append(dataR.DATE_RESILIATION[4])
                for i in range(dataR.Count[5]):
                        dataKK.append(dataR.DATE_RESILIATION[5])
                for i in range(dataR.Count[6]):
                        dataKK.append(dataR.DATE_RESILIATION[6])
                for i in range(dataR.Count[7]):
                        dataKK.append(dataR.DATE_RESILIATION[7])
                for i in range(dataR.Count[8]):
                        dataKK.append(dataR.DATE_RESILIATION[8])
                for i in range(dataR.Count[9]):
                        dataKK.append(dataR.DATE_RESILIATION[9])
                for i in range(dataR.Count[10]):
                        dataKK.append(dataR.DATE_RESILIATION[10])
                for i in range(dataR.Count[11]):
                        dataKK.append(dataR.DATE_RESILIATION[11])
                dataVV = pd.DataFrame()
                dataVV['DATE_RESILIATION']=dataKK
                dataVV['Fraud']=dff[maskk]["Fraud"]
                df_grouped7 = dataVV.groupby(by=['DATE_RESILIATION']).count()[['Fraud']]
                df_grouped7 = df_grouped7.reset_index()
                bar_chart77 = px.bar(df_grouped7,
                        x='DATE_RESILIATION',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#1a9107']*len(df_grouped7),
                        template= 'plotly_white')
                st.plotly_chart(bar_chart77)


            with col109:
                dataM = pd.DataFrame({'dateEffetPolice': ["2015-01-01","2017-01-01","2016-01-01","2009-01-01","2010-01-01","2012-01-02","2017-01-03","2016-11-01","2016-10-06","2016-11-10","2016-10-05","2016-11-15"], 'Count': [155,126,110,69,60,49,46,45,43,43,41,40]})
                dataK=[]
                for i in range(dataM.Count[0]):
                    dataK.append(dataM.dateEffetPolice[0])
                for i in range(dataM.Count[1]):
                    dataK.append(dataM.dateEffetPolice[1])
                for i in range(dataM.Count[2]):
                    dataK.append(dataM.dateEffetPolice[2])
                for i in range(dataM.Count[3]):
                    dataK.append(dataM.dateEffetPolice[3])
                for i in range(dataM.Count[4]):
                    dataK.append(dataM.dateEffetPolice[4])
                for i in range(dataM.Count[5]):
                    dataK.append(dataM.dateEffetPolice[5])
                for i in range(dataM.Count[6]):
                    dataK.append(dataM.dateEffetPolice[6])
                for i in range(dataM.Count[7]):
                    dataK.append(dataM.dateEffetPolice[7])
                for i in range(dataM.Count[8]):
                    dataK.append(dataM.dateEffetPolice[8])
                for i in range(dataM.Count[9]):
                    dataK.append(dataM.dateEffetPolice[9])
                for i in range(dataM.Count[10]):
                    dataK.append(dataM.dateEffetPolice[10])
                for i in range(dataM.Count[11]):
                    dataK.append(dataM.dateEffetPolice[11])
                dataV = pd.DataFrame()
                dataV['dateEffetPolice']=dataK
                dataV['Fraud']=dff[maskk]["Fraud"]
                df_grouped10 = dataV.groupby(by=['dateEffetPolice']).count()[['Fraud']]
                df_grouped10 = df_grouped10.reset_index()
                bar_chart999 = px.bar(df_grouped10,
                        x='dateEffetPolice',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#1a9107']*len(df_grouped10),
                        template= 'plotly_white')
                st.plotly_chart(bar_chart999)



            col106,col107=st.columns(2)
            with col106:

                dataMM = pd.DataFrame({'time': ["1H","11H","16H","14H","12H05","12H","12H03","12H12","15H","12H07","12H03"], 'Count': [263,235,190,187,181,180,180,180,178,174,174]})
                dataKKK=[]
                for i in range(dataMM.Count[0]):
                    dataKKK.append(dataMM.time[0])
                for i in range(dataMM.Count[1]):
                    dataKKK.append(dataMM.time[1])
                for i in range(dataMM.Count[2]):
                    dataKKK.append(dataMM.time[2])
                for i in range(dataMM.Count[3]):
                    dataKKK.append(dataMM.time[3])
                for i in range(dataMM.Count[4]):
                    dataKKK.append(dataMM.time[4])
                for i in range(dataMM.Count[5]):
                    dataKKK.append(dataMM.time[5])
                for i in range(dataMM.Count[6]):
                    dataKKK.append(dataMM.time[6])
                for i in range(dataMM.Count[7]):
                    dataKKK.append(dataMM.time[7])
                for i in range(dataMM.Count[8]):
                    dataKKK.append(dataMM.time[8])
                for i in range(dataMM.Count[9]):
                    dataKKK.append(dataMM.time[9])
                for i in range(dataMM.Count[10]):
                    dataKKK.append(dataMM.time[10])
                dataVVV = pd.DataFrame()
                dataVVV['time']=dataKKK
                dataVVV['Fraud']=dff[maskk]["Fraud"]
                df_grouped8 = dataVVV.groupby(by=['time']).count()[['Fraud']]
                df_grouped8 = df_grouped8.reset_index()
                bar_chart88 = px.bar(df_grouped8,
                        x='time',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#074f91']*len(df_grouped8),
                        template= 'plotly_white')
                st.plotly_chart(bar_chart88)

            with col107:

                dataN = pd.DataFrame({'time': ["Tunis","Ariana","Sousse","Sfax","Neuble","Monstir","Marsa","Bardo","Hammamet","Ben Arous","Gabes","Bizert","Tatouine","Soukra","Kairouan"], 'Count': [4002,962,820,739,234,285,163,99,98,234,87,73,71,64,34]})
                dataL=[]
                for i in range(dataN.Count[0]):
                    dataL.append(dataN.time[0])
                for i in range(dataN.Count[1]):
                    dataL.append(dataN.time[1])
                for i in range(dataN.Count[2]):
                    dataL.append(dataN.time[2])
                for i in range(dataN.Count[3]):
                    dataL.append(dataN.time[3])
                for i in range(dataN.Count[4]):
                    dataL.append(dataN.time[4])
                for i in range(dataN.Count[5]):
                    dataL.append(dataN.time[5])
                for i in range(dataN.Count[6]):
                    dataL.append(dataN.time[6])
                for i in range(dataN.Count[7]):
                    dataL.append(dataN.time[7])
                for i in range(dataN.Count[8]):
                    dataL.append(dataN.time[8])
                for i in range(dataN.Count[9]):
                    dataL.append(dataN.time[9])
                for i in range(dataN.Count[10]):
                    dataL.append(dataN.time[10])
                for i in range(dataN.Count[11]):
                    dataL.append(dataN.time[11])
                for i in range(dataN.Count[12]):
                    dataL.append(dataN.time[12])
                for i in range(dataN.Count[13]):
                    dataL.append(dataN.time[13])
                for i in range(dataN.Count[14]):
                    dataL.append(dataN.time[14])
                dataO = pd.DataFrame()
                dataO['localisation']=dataL
                dataO['Fraud']=dff[maskk]["Fraud"]
                df_grouped9 = dataO.groupby(by=['localisation']).count()[['Fraud']]
                df_grouped9 = df_grouped9.reset_index()
                bar_chart99 = px.bar(df_grouped9,
                        x='localisation',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#022d54']*len(df_grouped9),
                        template= 'plotly_white')
                st.plotly_chart(bar_chart99)


            dfmap = pd.DataFrame({'latitude':[35.672798,36.830872,36.812790,36.741910,36.769160,33.850388,36.397641,36.893160,35.759451,36.431469,34.712761,36.874898,35.796896,31.705825,36.799060],'longitude':[10.094910,10.097370,10.140420,10.229150,10.100590,10.095099,10.606924,10.318007,10.814300,10.732960,10.720964,10.258519,10.623109,9.895033,10.181551]})
            st.map(dfmap)


            colm3,colm4=st.columns(2)
            with colm3:
                dataMMMM = pd.DataFrame({'date': ["2017-05-08","2017-04-17","2017-04-11","2017-02-20","2017-03-06","2017-04-24","2017-04-21","2017-05-02","2017-03-08","2017-03-07","2017-03-30","2017-04-03"], 'Count': [126,119,113,107,104,104,104,104,101,100,99,97]})
                dataKKKKK=[]
                for i in range(dataMMMM.Count[0]):
                    dataKKKK.append(dataMMMM.date[0])
                for i in range(dataMMMM.Count[1]):
                    dataKKKK.append(dataMMMM.date[1])
                for i in range(dataMMMM.Count[2]):
                    dataKKKK.append(dataMMMM.date[2])
                for i in range(dataMMMM.Count[3]):
                    dataKKKK.append(dataMMMM.date[3])
                for i in range(dataMMMM.Count[4]):
                    dataKKKK.append(dataMMMM.date[4])
                for i in range(dataMMMM.Count[5]):
                    dataKKKK.append(dataMMMM.date[5])
                for i in range(dataMMMM.Count[6]):
                    dataKKKK.append(dataMMMM.date[6])
                for i in range(dataMMMM.Count[7]):
                    dataKKKK.append(dataMMMM.date[7])
                for i in range(dataMMMM.Count[8]):
                    dataKKKK.append(dataMMMM.date[8])
                for i in range(dataMMMM.Count[9]):
                    dataKKKK.append(dataMMMM.date[9])
                for i in range(dataMMMM.Count[10]):
                    dataKKKK.append(dataMMMM.date[10])
                for i in range(dataMMMM.Count[11]):
                    dataKKKK.append(dataMMMM.date[11])
                dataVVVVV = pd.DataFrame()
                dataVVVVV["trop_d'accidents"]=dataKKKK
                dataVVVVV['Fraud']=dff[maskk]['Fraud']
                df_grouped99 = dataVVVVV.groupby(by=["trop_d'accidents"]).count()[['Fraud']]
                df_grouped99 = df_grouped99.reset_index()
                bar_chart9999 = px.bar(df_grouped99,
                        x="trop_d'accidents",
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#91044d']*len(df_grouped99),
                        template= 'plotly_white')
                st.plotly_chart(bar_chart9999)

            with colm4:
                def load_lottiefile(filepath: str):
                    with open(filepath, "r") as f:
                        return json.load(f)
                
                def load_lottieurl(url: str):
                    r = requests.get(url)
                    if r.status_code != 200:
                        return None
                    return r.json()

                lottie_codingacc= load_lottiefile("coding2.json")
                lottie_acc=load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_uh95124g.json")
                st_lottie(
                    lottie_codingacc,
                    speed=1,
                    reverse=False,
                    loop=False,
                    quality="low",
                    height=0.5,
                    width=0.1,
                    key=None,
                )
                st_lottie(lottie_acc,key="acc")


            colm1,colm2=st.columns(2)
            with colm1:
                df_groupedpr = dff[maskk].groupby(by=['pourcentadeDeResponsabilite']).count()[['Fraud']]
                df_groupedpr = df_groupedpr.reset_index()
                bar_chartpr = px.bar(df_groupedpr,
                        x='pourcentadeDeResponsabilite',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#a30796']*len(df_groupedpr),
                        template= 'plotly_white')
                st.plotly_chart(bar_chartpr)

            with colm2:
                df_groupedns = dff[maskk].groupby(by=['natureDuSinistre']).count()[['Fraud']]
                df_groupedns = df_groupedns.reset_index()
                bar_chartns = px.bar(df_groupedns,
                        x='natureDuSinistre',
                        y='Fraud',
                        text='Fraud',
                        color_discrete_sequence = ['#aab58f']*len(df_groupedns),
                    template= 'plotly_white')
                st.plotly_chart(bar_chartns)
    else:
        st.error("Erreur votre nom d'utilisateur ou mot de passe est incorrect !!")