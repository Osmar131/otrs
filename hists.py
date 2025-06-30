import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io

def deploy_hist_col1(image, image_rgb, image_eq, channels, show_histograms):
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
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
        
def deploy_hist_col2(image, image_rgb, gray_image_eq, image_eq, channels, show_channels):
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
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
    # Canales de color
    channels = ['Red', 'Green', 'Blue']
    cmaps = ['Reds', 'Greens', 'Blues']
    colors = ['r', 'g', 'b']
        
    if show_channels:
        img_channels_rgb = cv2.hconcat([image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]])
        img_channels_eq = cv2.hconcat([image_eq[:, :, 0], image_eq[:, :, 1], image_eq[:, :, 2]])
        st.markdown("### 🔴🟢🔵 RGB Channel Analysis")
        fig_orig = plt.figure()
        for i, (color, channel) in enumerate(zip(colors, channels), 1):
            plt.subplot(1, 3, i)
            im_red = image_rgb[:, :, 0]
            plt.imshow(image_rgb[:, :, i-1], cmap='gray') # , cmap=cmaps[i-1]
            plt.title(f'Channel {channel}')
            plt.axis('off')
        st.pyplot(fig_orig)

        # Ecualizado
        st.markdown("#### Equalized Channel Visualization")
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
