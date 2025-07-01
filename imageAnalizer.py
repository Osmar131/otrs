import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
import hists
import convol_function

# Función para ecualización en cada canal
def equalize_rgb(image, clip_limit):
    # Convertir a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Aplicar CLAHE al canal V (Value/Brillo)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    v_equalized = clahe.apply(v)

    # Recombinar canales
    hsv_eq = cv2.merge([h, s, v_equalized])
    rgb_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    return rgb_eq

## Este código solo se ejecuta si el archivo se corre directamente
if __name__ == "__main__":

    # Configuración de la página (wide mode y barra lateral colapsada por defecto)
    st.set_page_config(page_title="Pixel-Level Image Analysis",
                       page_icon="📊",
                       layout="wide",
                       initial_sidebar_state="collapsed")

    # Título principal con estilo
    st.markdown("""
        <style>
        .big-font {
            font-size:28px !important;
            font-weight: bold;
            color: #4a4a4a;
        }
        </style>
        <p class="big-font">🎨 Image Processing Tools Histogram equalization, contrast adjustment, and more</p>
        """, unsafe_allow_html=True)

    # Imagen por defecto
    default_image_path = "Pic_0698_or06158.png"  # Cambia esto a tu imagen local
    default_image = Image.open(default_image_path)

    # Widget para cargar la imagen
    uploaded_file = st.file_uploader("Upload your image",
                                     type=["jpg", "jpeg", "png", "webp"],
                                     help="Supported formats: JPG, PNG, WEBP")

    # Mostrar imagen (cargada o por defecto)
    if uploaded_file is not None:
        # Leer la imagen subida
        image = Image.open(uploaded_file)
        st.success("Image uploaded successfully!")
    else:
        image = default_image
        st.info("🖼️ Using default sample image")

    ##image = Image.open(uploaded_file)
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sidebar con controles
    with st.sidebar:
        # Selectbox
        opcion = st.selectbox(
            "Select a category",
            ("RGB Image", "Gray Image", "Smoothing", "Filtered", "Edges", "Others"))

        if "RGB Image" == opcion:
            num_chann = 3
            st.header("⚙️ ⚖️ Image Equalization Settings")
            clip_limit = st.slider("Contrast Limit (CLAHE)", 1.0, 5.0, 2.0, 0.1)
            show_histograms = st.checkbox("📊 Display histograms", True)
            show_channels = st.checkbox("🌈 Show individual color channels", False)
            # Ecualización
            image_eq = equalize_rgb(image_rgb, clip_limit)
            image_eq = cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)
            gray_image_eq = image_eq
        if "Gray Image" == opcion:
            num_chann = 1
            st.header("⚙️️ ⚖️ Image Equalization Settings")
            clip_limit = st.slider("Contrast Limit (CLAHE)", 1.0, 5.0, 2.0, 0.1)
            show_histograms = st.checkbox("📊 Display histograms", False)
            show_channels = st.checkbox("🌈 Show individual color channels", False)
            # Ecualización
            image_eq = equalize_rgb(image_rgb, clip_limit)
            image_eq = cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)
            gray_image_eq = cv2.cvtColor(image_eq, cv2.COLOR_RGB2GRAY)
            image = gray_image
        if "Filtered":
            # Selector de tamaño de kernel (solo valores impares)
            st.header("⚙️️ ⚖️ Depending on size of image, it could take time to process images")
            kernel_size = st.selectbox(
                "Tamaño del Kernel (Solo impares)",
                options=[3, 5, 7, 9, 11],
                index=0,
                help="El tamaño debe ser impar para tener un centro definido"
            )
            
            # Control deslizante para el multiplicador
            neighbor_multiplier = st.slider(
                "Neighor multiplier",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Controls how much neighboring pixels affect the result"
            )
            gray_convolution = st.checkbox("Convolution gray image", False)

    # Diseño de columnas
    col1, col2  = st.columns([1,1])
    if 'RGB Image' == opcion:
    ##    deploy_histograma(image, image_eq, num_chann)
        col1, col2  = st.columns([1,1])
        with col1:
            value = hists.deploy_hist_col1(image, image_rgb, image_eq, num_chann, show_histograms)
        with col2:
            value = hists.deploy_hist_col2(image, image_rgb, gray_image_eq, image_eq, num_chann, show_channels)

    elif 'Gray Image' == opcion:
    ##    deploy_histograma(image, image_eq, num_chann)
        with col1:
            value1 = hists.deploy_hist_col1(image, image_rgb, image_eq, num_chann, show_histograms)
        with col2:
            value2 = hists.deploy_hist_col2(image, image_rgb, gray_image_eq,  image_eq, num_chann, show_channels)
            
    elif 'Filtered' == opcion:
        if gray_convolution:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        with col1:
            st.header("Original Image (Before Processing)")
            st.image(image)
        with col2:
            img = np.array(image)
            value2 = convol_function.adjustable_convolution(img,
                                neighborhood_size = kernel_size,
                                neighbor_multiplier = neighbor_multiplier,
                                padding = 'same')
        st.header("Working on it...")
    elif 'Smoothing' == opcion:
        st.header("Working on it...")
    elif 'Other' == opcion:
        st.header("Working on it...")

