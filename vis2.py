import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests
import seaborn as sns
#+++++++++++++++

#page
st.set_page_config(page_title="Projet Mhamed",
page_icon=":bar_chart:",layout="wide")

#dataset
dff=pd.read_csv('algo.csv')
dff=dff.drop(['Unnamed: 0'],axis=1)

#tbadél les cors el liste
fraudd= dff['Fraud'].unique().tolist()
codeCompagniee=dff['codeCompagnie'].unique().tolist()
naturePolicee=dff['natureDuSinistre'].unique().tolist()


#parti ajnab
#sidebar
st.sidebar.header("please filter here:")

fraudd=st.sidebar.multiselect(
    "select type de Fraude:",
    options=fraudd,
    default=fraudd)

codeCompagniee=st.sidebar.multiselect(
    "select code de la compagnie:",
    options=codeCompagniee,
    default=codeCompagniee)

naturePolicee=st.sidebar.multiselect(
    "select la nature de sinistre:",
    options=naturePolicee,
    default=naturePolicee)

st.sidebar.write('---------------------------')
st.sidebar.write('Fraud 0: corrspond a client pas fraude')
st.sidebar.write('Fraud 1: corrspond a client fraude')
st.sidebar.write('---------------------------')
st.sidebar.write('code compagnie 1 : corrspond a le code de la societe Star')
st.sidebar.write('code compagnie 2 : corrspond a le code de la societe Gear')
st.sidebar.write('code compagnie 3 : corrspond a le code de la societe MAE')
st.sidebar.write('---------------------------')
st.sidebar.write('Nature de sinistre : "M" corrspond a un accident MATRIEL, "C" corrspond a un accident CORPOREL')

#end parti 
blank,tit,blankk= st.columns(3)
tit.title(":tractor: Analyse D'assurance :car:")
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
    st.subheader("CODE COMPAGNIE")
    st.subheader(f"{code:} {office}")

with col2:
    st.subheader("NOMBRES DES TRANSACTIONS")
    st.subheader(f"{totall:} {stars}")

with col3:
    st.subheader("NATURE DU SINISTRE")
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
    dataNB = pd.DataFrame({'dateEffetPolice': ["2015-01-01","2017-01-01","2016-01-01","2009-01-01","2010-01-01","2012-01-02","2017-01-03","2016-11-01","2016-10-06","2016-11-10","2016-10-05","2016-11-15"], 'Count': [155,126,110,69,60,49,46,45,43,43,41,40]})
    dataNBB=[]
    for i in range(dataNB.Count[0]):
        dataNBB.append(dataNB.dateEffetPolice[0])
    for i in range(dataNB.Count[1]):
        dataNBB.append(dataNB.dateEffetPolice[1])
    for i in range(dataNB.Count[2]):
        dataNBB.append(dataNB.dateEffetPolice[2])
    for i in range(dataNB.Count[3]):
        dataNBB.append(dataNB.dateEffetPolice[3])
    for i in range(dataNB.Count[4]):
        dataNBB.append(dataNB.dateEffetPolice[4])
    for i in range(dataNB.Count[5]):
        dataNBB.append(dataNB.dateEffetPolice[5])
    for i in range(dataNB.Count[6]):
        dataNBB.append(dataNB.dateEffetPolice[6])
    for i in range(dataNB.Count[7]):
        dataNBB.append(dataNB.dateEffetPolice[7])
    for i in range(dataNB.Count[8]):
        dataNBB.append(dataNB.dateEffetPolice[8])
    for i in range(dataNB.Count[9]):
        dataNBB.append(dataNB.dateEffetPolice[9])
    for i in range(dataNB.Count[10]):
        dataNBB.append(dataNB.dateEffetPolice[10])
    for i in range(dataNB.Count[11]):
        dataNBB.append(dataNB.dateEffetPolice[11])
    dataVZ = pd.DataFrame()
    dataVZ['dateEffetPolice']=dataNBB
    dataVZ['Fraud']=dff[maskk]["Fraud"]
    df_grouped01 = dataVZ.groupby(by=['dateEffetPolice']).count()[['Fraud']]
    df_grouped01 = df_grouped01.reset_index()
    bar_chart999 = px.bar(df_grouped01,
            x='dateEffetPolice',
            y='Fraud',
            text='Fraud',
            color_discrete_sequence = ['#1a9107']*len(df_grouped01),
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
    dataVVVVV['trop_accidents']=dataKKKK
    dataVVVVV['Fraud']=dff[maskk]['Fraud']
    df_grouped99 = dataVVVVV.groupby(by=['trop_accidents']).count()[['Fraud']]
    df_grouped99 = df_grouped99.reset_index()
    bar_chart9999 = px.bar(df_grouped99,
            x='trop_accidents',
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