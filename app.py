import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import requests
import gzip

@st.cache_resource
def load_models():
    model = pkl.load(open('model.pkl', 'rb'))
    cf_model = pkl.load(open('cf_model.pkl', 'rb'))
    return model, cf_model

@st.cache_data
def load_data():
    X = pkl.load(open('X.pkl', 'rb'))
    with gzip.open('Y.pkl.gz', 'rb') as f:
        Y = pkl.load(f)
    anime_names = pkl.load(open('anime_names.pkl', 'rb'))
    anime_id_df = pkl.load(open('anime_id_df.pkl', 'rb'))
    return X, Y, anime_names, anime_id_df

model, cf_model = load_models()
X, Y, anime_names, anime_id_df = load_data()

def fetch_poster(anime_id):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'data' in data:
            images = data['data']['images']
            image_url = images['jpg']['image_url']
            mal_url = data['data']['url']
            return image_url, mal_url
        else:
            return "https://wallpapers.com/images/high/anime-characters-1080-x-1920-picture-3ztfmcx0w79qi8p5.webp", "https://myanimelist.net/"
    except Exception as e:
        st.error(f"Error fetching poster: {e}")
        return "https://wallpapers.com/images/high/anime-characters-1080-x-1920-picture-3ztfmcx0w79qi8p5.webp", "https://myanimelist.net/"

def collaborative_recommend(anime):
    anime_id = anime_names[anime_names.Name == anime]['anime_id'].values[0]
    distances, indices = cf_model.kneighbors(Y.loc[anime_id].to_numpy().reshape(1, -1))

    animes = []
    anime_ids = []
    for i in indices[0][1:6]:  # Get top 5 recommendations
        animes.append(anime_id_df[anime_id_df.anime_id == Y.index[i]].Name.values[0])
        anime_ids.append(Y.index[i])

    return animes, anime_ids

def content_recommend(anime_name):
    anime_index = anime_names[anime_names.Name == anime_name].index.values[0]
    distances, indices = model.kneighbors(X[anime_index])

    animes = []
    anime_ids = []
    for i in indices[0][1:6]:  # Get top 5 recommendations
        animes.append(anime_names.iloc[i].Name)
        anime_ids.append(anime_names.iloc[i].anime_id)

    return animes, anime_ids

st.title('What anime to watch next?')
st.write('Anime recommendation system: collaborative filtering + content-based filtering.')
st.caption('*-by Md Faisal.*')

st.write('')
st.write('')
st.write('')
anime_name = st.selectbox('Choose your favourite anime.', anime_names['Name'], placeholder='try doraemon, naruto, ninja hattori, etc.')

if st.button('Recommend'):
    if anime_name:
        st.write("Content Based Recommendations: ")
        similar_animes, similar_anime_ids = content_recommend(anime_name)
        cols = st.columns(5)
        for col, anime, anime_id in zip(cols, similar_animes, similar_anime_ids):
            with col:
                st.caption(anime)
                image_url, mal_url = fetch_poster(anime_id)
                if image_url:
                    st.markdown(f"[![{anime}]({image_url})]({mal_url})")
        st.write('')
        st.divider()
        st.write("Users Also Liked (Collaborative Recommendations): ")
        similar_animes, similar_anime_ids = collaborative_recommend(anime_name)
        cols = st.columns(5)
        for col, anime, anime_id in zip(cols, similar_animes, similar_anime_ids):
            with col:
                st.caption(anime)
                image_url, mal_url = fetch_poster(anime_id)
                if image_url:
                    st.markdown(f"[![{anime}]({image_url})]({mal_url})")
    else:
        st.error('Please select an anime to get recommendations.')

st.sidebar.write("**Connect with me.**")
st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/md-fsl?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30">
</a> &nbsp; &nbsp;
<a href="https://github.com/Muhammed-Faisal" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30">
</a> &nbsp; &nbsp;
<a href="https://www.kaggle.com/mdfaisal1" target="_blank">
<img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" width="30" height="30">
</a>
""", unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("**Have any suggestions?**")
st.sidebar.write("*Please do let me know at mdf1234786143@gmail.com*")
