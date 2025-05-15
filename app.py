import pandas as pd
import pickle
import streamlit as st

# Cargar modelo y variables
filename = 'RandomForest.pkl'
modelRF, labelencoder, variables = pickle.load(open(filename, 'rb'))

# Estilos personalizados tipo Spotify
st.set_page_config(page_title="Spotify Recommender", layout = "centered")

# CSS para estilo Spotify
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stSlider > div {
            color: #1DB954; /* verde Spotify */
        }
        .stSlider .css-14pt78w, .stSlider .css-1l6qdhw {
            color: white;
        }
        .stSelectbox > div {
            color: white;
        }
        .stApp {
            background-color: #121212;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html = True)

# Imagen/logo
st.image("spotify_logo.png", width = 200)

# Título
st.title('🎧 Recomendador de Canciones')

# Inputs
daceability = st.slider('Bailabilidad 💃', 0.0, 1.0, 0.5, 0.1)
loudness = st.slider('Ruido 🔊', -60, 0, -30, 1)
speechiness = st.slider('Contenido hablado 🗣', 0.0, 1.0, 0.5, 0.1)
valence = st.slider('Valencia 😄', 0.0, 1.0, 0.5, 0.1)
tempo = st.slider('Tempo ⌛️', 0, 200, 100, 5)
duration = st.slider('Duración en ms 🕑', 0, 500000, 120000, 1000)
time_signature = st.selectbox('Compás 🎼', ['1', '3', '4', '5'])

# Crear DataFrame de entrada
datos = [[daceability, loudness, speechiness, valence, tempo, time_signature]]
data = pd.DataFrame(datos, columns = ['daceability', 'loudness', 'speechiness', 'valence', 'tempo', 'duration', 'time_signature'])

# Reordenar columnas
data_preparada = data.copy()
data_preparada = data_preparada.reindex(columns = variables, fill_value=0)

# Predicción
Y_fut = modelRF.predict(data_preparada)
data['⚡️ Resultado'] = labelencoder.inverse_transform(Y_fut)

# Resultado
st.subheader('Resultado de la Recomendación:')
st.dataframe(data, use_container_width = True)