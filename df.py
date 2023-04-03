from select import select
from ssl import Options
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import pickle
import datetime
#----importation Matricule------
from matplotlib import pyplot as plt
import imutils
import easyocr
import random
import cv2
import numpy as np
import plotly.express as px
#Parti Analyse sentimental
from streamlit_lottie import st_lottie
import json
import requests


#page
st.set_page_config(page_title="Projet Mhamed",
page_icon=":bar_chart:",
layout="wide")

#login
#st.title("information d'application")
st.header("Introduction de projet")
st.markdown("Ce site est destiné à l’MAE, il est créé pour l'administrateur, il contient quatre interfaces, chacune avec sa propre fonction, ce site intelligent résoudre certains problèmes de l'entreprise, afin de l’aider à la développer à travers l’intégration de l’intelligence artificielle, l’apprentissage automatique et l’intelligence d’affaires,  notamment la partie de la  détection de fraude des clients dans l'entreprise nous permet de détecter si le client est fraude ou non à travers le remplissage d’un formulaire,  ainsi que l'affichage des informations sur un véhicule à travers télécharger son image, ainsi que nous avons créés une analyse sentimental visuelle pour les avis des clients de notre entreprise et l'afficher sous forme d'une analyse graphique, enfin nous avons créé une analyse complète pour les assureurs et les compagnies d’assurances à travers des graphes et des analyses visuelles qui nous aident à améliorer l’entreprise, dans le but de faciliter la prise de décision et réduire leurs défauts.")
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


image = Image.open('MAE.png')
st.sidebar.image(image, width=300)
username=st.sidebar.text_input("Nom d'utilisateur")
password= st.sidebar.text_input("Mot de passe",type='password')
if st.sidebar.checkbox("Connexion"):
    if username=="mhamed" and password=="bou123":
        st.success("Connecté en tant que {}".format(username))

#Head
        selected = option_menu(
            menu_title=None,
            options=['DÉTECTION DE FRAUDE','DÉTECTION DE MATRICULE','ANALYSE SENTIMENTALE',"ANALYSE D'ASSURANCE"],
            icons = ['tv','person','heart'],
            menu_icon="cast",
            default_index = 0,
            orientation="horizontal",)


        #database 
        data=pd.read_csv('algo.csv')
        data=data.drop(['Unnamed: 0'],axis=1)
        #Machine learning
        dataml=pd.read_csv('ML.csv')
        dataml=dataml.drop(['Unnamed: 0'],axis=1)
        #importation des bébliothques
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from xgboost import XGBClassifier
        #traitement
        X=dataml.drop(columns=["identificationTiers","libUsage","jour_dateEcheancePolice","mois_dateEcheancePolice","VolontaireAccidents","CountSinisterAdvr","typeIntermediaire","vehicule_id","assure_id","CONTRAT_EN_COURS","pourcentadeDeResponsabilite","porcentageCompagnieAdverse","Duree","Fraud"])
        Y=dataml['Fraud']
        X_train, X_test, y_train, y_test = train_test_split(X.values,Y.values, test_size = 0.2, random_state = 1)
        xgb_clfdf = XGBClassifier()
        xgb_clfdf.fit(X_train, y_train)

        #Parti Detection de fraude
        if (selected == 'DÉTECTION DE FRAUDE'):
            #page title 
            st.title('Prédiction de Fraude dans la secteur assurance')

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

    else:
        st.error("Erreur votre nom d'utilisateur ou mot de passe est incorrect !!")