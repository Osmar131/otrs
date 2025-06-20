import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página (wide mode y barra lateral colapsada por defecto)
st.set_page_config(
    page_title="Análisis de Imágenes Pro",
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
    <p class="big-font">🎨 Procesamiento de Imágenes + Histograma</p>
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

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Controles de Ecualización")
    clip_limit = st.slider("Límite de contraste (CLAHE)", 1.0, 5.0, 2.0, 0.1)
    show_histograms = st.checkbox("Mostrar histogramas", True)
    show_channels = st.checkbox("Mostrar canales separados", False)

# Widget para cargar la imagen
uploaded_file = st.file_uploader(
    "Sube una imagen", 
    type=["jpg", "jpeg", "png", "webp"],
    help="Formatos soportados: JPG, PNG, WEBP"
)

image = Image.open(uploaded_file)
image = np.array(image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Ecualización
image_eq = equalize_rgb(image_rgb, clip_limit)

# Diseño de columnas
col1, col2  = st.columns([1,1])
with col1:
    st.header("Imagen original")
    st.image(image)#, width=width_img)

    if show_histograms:
        st.markdown("### 📊 Histogramas Comparativos")
        # Crear figura con subplots
        fig, axes = plt.subplots(3,2)

        # Canales de color
        channels = ['Rojo', 'Verde', 'Azul']
        colors = ['r', 'g', 'b']

        for i, (color, channel) in enumerate(zip(colors, channels)):
            # Histogramas original
            axes[i, 0].hist(image_rgb[:,:, i].ravel(), 256, [0,256], color=color, alpha=0.7)
            axes[i, 0].set_title(f'Original - Canal {channel}')
            axes[i, 0].grid(alpha=0.3)
            # Histogramas ecualizado
            axes[i, 1].hist(image_eq[:, :, i].ravel(), 256, [0,256], color=color, alpha=0.7)
            axes[i, 1].set_title(f'Ecualizado - Canal {channel}')
            axes[i, 1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
    
    with st.expander("📊 Métricas", expanded=True):
        st.metric("Brillo Medio", f"{np.mean(image_rgb):.1f}")
        st.progress(np.mean(image_rgb)/255)
        st.metric("Contraste (Desviación Estándar)", f"{np.std(image_rgb):.1f}")
        
with col2:
    st.header("Imagen equalizada")
    st.image(image_eq)#, width=width_img)
    if show_channels:
        st.markdown("### 🖼️ Canales de Color")
        fig_orig = plt.figure()
        for i, (color, channel) in enumerate(zip(colors, channels), 1):
            plt.subplot(1, 3, i)
            plt.imshow(image_rgb[:, :, i-1]) # , cmap='gray'
            plt.title(f'Canal {channel}')
            plt.axis('off')
        st.pyplot(fig_orig)

        # Ecualizado
        st.markdown("#### Ecualizado")
        fig_eq = plt.figure()
        for i, (color, channel) in enumerate(zip(colors, channels), 1):
            plt.subplot(1, 3, i)
            plt.imshow(image_eq[:, :, i-1]) # , cmap='gray'
            plt.title(f'Canal {channel}')
            plt.axis('off')
        st.pyplot(fig_eq)
##    st.header("Grafico Escala de grises")
##    fig, (ax, ax2) = plt.subplots(1,2)
##    ax.hist(gray_image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
##    ax.set_xlim([0, 256])
##    ax.set_xlabel("Intensidad de píxeles")
##    ax.set_ylabel("Frecuencia")
##    st.pyplot(fig)

    # Estadísticas debajo de la imagen
    with st.expander("📊 Métricas", expanded=True):
        st.metric("Brillo Medio", f"{np.mean(gray_image):.1f}")
        st.progress(np.mean(gray_image)/255)
        st.metric("Contraste (Desviación Estándar)", f"{np.std(gray_image):.1f}")

    # Histograma ecualizado
##    ax2.hist(image_eq.ravel(), 256, [0,256], color='blue', alpha=0.7)
##    ax2.set_title('Ecualizado (CLAHE)')
##    ax2.grid(alpha=0.2)





