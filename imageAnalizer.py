import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io

# Configuración de la página (wide mode y barra lateral colapsada por defecto)
st.set_page_config(
    page_title="Pixel-Level Image Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# Imagen por defecto
default_image_path = "Pic_0698_or06158.png"  # Cambia esto a tu imagen local
default_image = Image.open(default_image_path)

# Widget para cargar la imagen
uploaded_file = st.file_uploader(
    "Upload your image", 
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
        ("RGB Image", "Gray Image", "Smoothing", "Filtered", "Edges"))

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
        show_histograms = st.checkbox("📊 Display histograms", True)
        show_channels = st.checkbox("🌈 Show individual color channels", False)
        # Ecualización
        image_eq = equalize_rgb(image_rgb, clip_limit)
        image_eq = cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)
        gray_image_eq = cv2.cvtColor(image_eq, cv2.COLOR_RGB2GRAY)
        image = gray_image
        
def deploy_histograma(image, image_eq, channels):
    # Diseño de columnas
    col1, col2  = st.columns([1,1])
##        image = gray_image
##        image_eq = gray_image_eq

    with col1:
        st.header("🆚 Original Image (Before Processing)")
        st.image(image)#, width=width_img)

        if show_histograms:
            st.markdown("### 📊 Comparative Histograms")
            # Crear figura con subplots
            fig, axes = plt.subplots(3,2)

            # Canales de color
            channels = ['Red', 'Green', 'Blue']
            cmaps = ['Reds', 'Greens', 'Blues']
            colors = ['r', 'g', 'b']

            for i, (color, channel) in enumerate(zip(colors, channels)):
                # Histogramas original
                axes[i, 0].hist(image_rgb[:,:, i].ravel(), 256, [0,256], color=color, alpha=0.7)
                axes[i, 0].set_title(f'Original - Channel {channel}')
                axes[i, 0].grid(alpha=0.3)
                # Histogramas ecualizado
                axes[i, 1].hist(image_eq[:, :, i].ravel(), 256, [0,256], color=color, alpha=0.7)
                axes[i, 1].set_title(f'Equalized - Channel {channel}')
                axes[i, 1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
        
        with st.expander("📊 Metrics", expanded=True):
            st.metric("🔆 Mean Brightness Level", f"{np.mean(image_rgb):.1f}")
            st.progress(np.mean(image_rgb)/255)
            st.metric("⚖️ Contrast (σ)", f"{np.std(image_rgb):.1f}")
    
    with col2:
        st.header("🔄 Equalized Output")
        st.image(gray_image_eq)#, width=width_img)
        # Bóton de descarga
        _, img_encoded = cv2.imencode(".png", image_eq)
        img_byte_arr = img_encoded.tobytes()

        st.download_button(
            label="⬇️ Save Equalized Image",
            data=img_byte_arr,
            file_name="image.png",
            mime="image/png"
            )
        if show_channels:
            img_channels_rgb = cv2.hconcat([image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]])
            img_channels_eq = cv2.hconcat([image_eq[:, :, 0], image_eq[:, :, 1], image_eq[:, :, 2]])
            st.markdown("### 🖼 🔴🟢🔵 RGB Channel Analysis")
            fig_orig = plt.figure()
            for i, (color, channel) in enumerate(zip(colors, channels), 1):
                plt.subplot(1, 3, i)
                im_red = image_rgb[:, :, 0]
                plt.imshow(image_rgb[:, :, i-1], cmap='gray') # , cmap=cmaps[i-1]
                plt.title(f'Channel {channel}')
                plt.axis('off')
            st.pyplot(fig_orig)

            # Ecualizado
            st.markdown("#### 🌈 Equalized Channel Visualization")
            fig_eq = plt.figure()
            for i, (color, channel) in enumerate(zip(colors, channels), 1):
                plt.subplot(1, 3, i)
                plt.imshow(image_eq[:, :, i-1], cmap='gray') # , cmap=cmaps[i-1]
                plt.title(f'Channel {channel}')
                plt.axis('off')
            st.pyplot(fig_eq)

            _, img_encoded = cv2.imencode(".png", img_channels_eq)
            img_byte_arr = img_encoded.tobytes()

            st.download_button(
                label="⬇️ Save Equalized RGB Channels",
                data=img_byte_arr,
                file_name="image.png",
                mime="image/png"
                )

        # Estadísticas debajo de la imagen
        with st.expander("📊 Metrics", expanded=True):
            st.metric("🔆 Mean Brightness Level", f"{np.mean(gray_image):.1f}")
            st.progress(np.mean(gray_image)/255)
            st.metric("⚖️ Contrast (σ)", f"{np.std(gray_image):.1f}")

# Diseño de columnas
if 'RGB Image' == opcion:
    deploy_histograma(image, image_eq, num_chann)
elif 'Gray Image' == opcion:
    deploy_histograma(image, image_eq, num_chann)
elif 'Smoothing' == opcion:
    st.header("Working on it...")
elif 'Other' == opcion:
    st.header("Working on it...")

