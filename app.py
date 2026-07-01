import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIGURACIÓN VISUAL
# =========================
st.set_page_config(
    page_title="Clasificador de Flores AI",
    page_icon="🌸",
    layout="centered"
)

# =========================
# ESTILO CSS PERSONALIZADO
# =========================
st.markdown("""
<style>
body {
    background-color: #f7f7ff;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #6a1b9a;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.result {
    font-size: 24px;
    font-weight: bold;
    color: #2e7d32;
}

.confidence {
    font-size: 18px;
    color: #1565c0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITULO
# =========================
st.markdown('<div class="title">🌸 Clasificador de Flores AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube o toma una foto y descubre qué flor es 🌿</div>', unsafe_allow_html=True)

# =========================
# AUTOR
# =========================
st.markdown("""
<div class="card">
👩‍🎓 <b>Desarrollado por:</b><br>
Genesis Yuliana Medina Ramos<br>
📌 20231900117
</div>
""", unsafe_allow_html=True)

# =========================
# MODELO
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/flowers_model.keras")

model = load_model()

classes = [
    "Margarita",
    "Diente de león",
    "Rosa",
    "Girasol",
    "Tulipán"
]

# =========================
# INPUT
# =========================
option = st.radio("📌 Selecciona una opción:", ["Subir imagen", "Tomar foto"])

file = None

if option == "Subir imagen":
    file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("📸 Toma una foto")

# =========================
# PREDICCIÓN
# =========================
if file is not None:

    image = Image.open(file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Imagen seleccionada", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocesamiento
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    clase = classes[index]
    confidence = np.max(prediction) * 100

    # RESULTADO BONITO
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(f'<div class="result">🌸 Flor detectada: {clase}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">🎯 Confianza: {confidence:.2f}%</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # PROBABILIDADES
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Probabilidades")

    for i, c in enumerate(classes):
        st.write(f"{c}: {prediction[0][i]*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
---
<center>🌸 IA Clasificador de Flores | Proyecto de Inteligencia Artificial</center>
""", unsafe_allow_html=True)