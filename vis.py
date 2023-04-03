import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests
#+++++++++++++++

#page
st.set_page_config(page_title="Projet Mhamed",
page_icon=":bar_chart:",layout="wide")

#dataset
df=pd.read_csv('rslt.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df["sentiment"].replace({'Neutral':'Neutre'}, inplace=True)


#tbadél les cors el liste
Yearr= df['Year'].unique().tolist()
Monthh=df['Month'].unique().tolist()
sentimentt=df['sentiment'].unique().tolist()

#parti ajnab
#sidebar
st.sidebar.header("please filter here:")
sentimentss=st.sidebar.multiselect(
    "select sentiment svp:",
    options=sentimentt,
    default=sentimentt
)

anne=st.sidebar.multiselect(
    "select the Year:",
    options=Yearr,
    default=Yearr
)

moisss=st.sidebar.slider(
    'Select le Mois: ',
    min_value= min(Monthh),
    max_value= max(Monthh),
    value= (min(Monthh),max(Monthh))
)

#end parti 
blank,tit,blankk= st.columns(3)
tit.title(":angry: Analyse Sentimentale :smiley:")
st.markdown("##")



mask = (df['Month'].between(*moisss)) & (df['sentiment'].isin(sentimentss)) & (df['Year'].isin(anne))

#créat le number
total_comment = int(df[mask].shape[0])
yer=df[mask].Year.unique()
mo=df[mask].Month.unique()
stars= ":speech_balloon:"
datee= ":date:"

col1,col2,col3=st.columns(3)
with col1:
    st.markdown("L'ANNE")
    st.subheader(f"{yer:} {datee}")

with col2:
    st.markdown("NOMBRES DES COMMENTAIRES")
    #st.metric(label="NOMBRES DES COMMENTAIRES", value=total_comment, delta="1.2 °F")
    st.subheader(f"{total_comment:} {stars}")

with col3:
    st.subheader("MOIS")
    st.markdown(f"{mo:} {datee}")

st.markdown("---")

#streamlite selection


col4,col5,col6=st.columns(3)
with col4:
    monthsel= st.slider('Mois: ',
    min_value= min(Monthh),
    max_value= max(Monthh),
    value= (min(Monthh),max(Monthh)))
with col5:
    month_sel=st.multiselect('Sentiment',
    sentimentt,
    default=sentimentt)
with col6:
    rs=st.selectbox("Years select", Yearr)

st.markdown("---")


#st.markdown(f'*Available Results: {number_of_result}*')

# --- GROUP DATAFRAME AFTER SELECTION
df_grouped = df[mask].groupby(by=['sentiment']).count()[['Month']]
df_grouped = df_grouped.reset_index()

col7,col8=st.columns(2)
with col7:
    bar_chart = px.bar(df_grouped,
                   x='sentiment',
                   y='Month',
                   text='Month',
                   color_discrete_sequence = ['#297694']*len(df_grouped),
                   template= 'plotly_white')
    st.plotly_chart(bar_chart)

with col8:
    pie_chart = px.pie(df[mask],title="",values="Year",names="sentiment")
    st.plotly_chart(pie_chart)

col9,col10=st.columns(2)
with col9:
    df_groupedT = df[mask].groupby(by=['Year']).count()[['Month']]
    df_groupedT = df_groupedT.reset_index()
    bar_chart2 = px.bar(df_groupedT,
        x='Year',
        y='Month',
        title="test",
        text='Month',
        color_discrete_sequence = ['#902994']*len(df_grouped),
        template= 'plotly_white')
    st.plotly_chart(bar_chart2)

with col10:
    df_groupedP = df[mask].groupby(by=['Month']).count()[['Comment']]
    df_groupedP = df_groupedP.reset_index()
    bar_chart3 = px.bar(df_groupedP,
        x='Month',
        y='Comment',
        text='Comment',
        color_discrete_sequence = ['#F63366']*len(df_grouped),
        template= 'plotly_white')
    st.plotly_chart(bar_chart3)

col13,col14,col15=st.columns(3)
with col13:
    st.dataframe(df[mask]["Comment"])

with col14:
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_coding= load_lottiefile("coding.json")
    lottie_hello=load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_vpjqslwa.json")
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=False,
        quality="low",
        height=0.5,
        width=0.1,
        key=None,
    )
    st_lottie(lottie_hello,key="hello")


with col15:
    fig4=plt.figure()
    df[mask][df.sentiment=='Positive'].Year.plot(kind="kde", label="Positive")
    df[mask][df.sentiment=='Negative'].Year.plot(kind="kde", label="Negative")
    df[mask][df.sentiment=='Neutre'].Year.plot(kind="kde", label="Neutre")
    plt.legend()
    st.pyplot(fig4)

st.markdown("---")


