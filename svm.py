# cd  C:\Users\quewa\Documents\pyzo\new
# streamlit run svm.py

import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler

filename = 'svm_model.sav'

# Chargez le modèle à partir du fichier
with open(filename, 'rb') as file:
    clf = pickle.load(file)

st.title("Dessinez un chiffre")

# Créez une zone de dessin
a="""
if st.button("Effacer"):
    image = np.zeros((8, 8), dtype=np.uint8)
else:
    image = st.session_state.get("image", np.zeros((8, 8), dtype=np.uint8))
"""
drawing = st.session_state.get("drawing", False)

if drawing:
    x, y = st.session_state["last_xy"]
    row, col = y // 25, x // 25
    image[row, col] = 255

def on_canvas_motion(event):
    st.session_state["last_xy"] = event.x, event.y
    st.session_state["drawing"] = True


from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=25,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    #on_move=on_canvas_motion,
    key="canvas",
)



# Affichez l'image en temps réel
if st.button("Prédire le chiffre"):
    im1=canvas_result.image_data
    red_matrix = im1[:, :, 0]
    im2=np.array(red_matrix)/255
    im3=1-im2
    im4=im3*16
    #print(im4)
    # Créez une nouvelle matrice de taille 8x8
    im5=np.zeros((8, 8), dtype=int)
    # Agrégez les valeurs de chaque bloc 4x4
    for i in range(8):
        for j in range(8):
            block = im4[i*25:(i+1)*25, j*25:(j+1)*25]
            im5[i, j] = np.mean(block)
    #print(block)
    #input_data = image.ravel() * 16
    #print(im5)
    im6=im5.reshape(1,64)
    #st.write(im6)
    
    prediction = clf.predict(im6)
    st.write("Prédiction : ", prediction[0])


