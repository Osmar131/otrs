import streamlit as st
import cv2
import numpy as np
from typing import Union, Tuple

def adjustable_convolution(
    image: np.ndarray,
    neighborhood_size: int = 3,
    neighbor_multiplier: float = 0.5,
    custom_kernel: Union[np.ndarray, None] = None,
    padding: Union[str, int] = 0,
    stride: int = 1
) -> np.ndarray:
    """
    Applies convolution with adjustable kernel and neighbor consideration.

    Args:
        image: Input image (2D for grayscale, 3D for RGB)
        neighborhood_size: Neighborhood window size (odd, e.g. 3, 5, 7)
        neighbor_multiplier: Weighting factor for neighboring pixels
        custom_kernel: Custom kernel (overrides neighborhood_size if provided)
        padding: 'same' to maintain size, or number of padding pixels
        stride: Kernel displacement step

    Returns:
        Convolved image (same size as input if padding='same')
    """
    # Validations
    if len(image.shape) not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")
    
    if custom_kernel is None:
        if neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be odd (3, 5, 7...)")
        
        # Base kernel (identity)
        kernel = np.zeros((neighborhood_size, neighborhood_size))
        center = neighborhood_size // 2
        kernel[center, center] = 1.0  # Center pixel
        
        # Add neighbor weighting
        for i in range(neighborhood_size):
            for j in range(neighborhood_size):
                if i != center or j != center:
                    distance = max(abs(i - center), abs(j - center))
                    kernel[i, j] = neighbor_multiplier / distance
    else:
        kernel = custom_kernel
        neighborhood_size = kernel.shape[0]

    # Padding handling
    if padding == 'same':
        pad = neighborhood_size // 2
    else:
        pad = int(padding)

    if pad > 0:
        if len(image.shape) == 3:  # RGB
            image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        else:  # Grayscale
            image = np.pad(image, pad, mode='constant')

    # Prepare output image
    h, w = image.shape[:2]
    out_h = (h - neighborhood_size) // stride + 1
    out_w = (w - neighborhood_size) // stride + 1
    
    if len(image.shape) == 3:
        output = np.zeros((out_h, out_w, image.shape[2]))
    else:
        output = np.zeros((out_h, out_w))

    # Convolution
    for c in range(output.shape[2]) if len(image.shape) == 3 else [None]:
        for y in range(0, out_h):
            for x in range(0, out_w):
                roi = image[
                    y*stride : y*stride + neighborhood_size,
                    x*stride : x*stride + neighborhood_size,
                    c if c is not None else ...
                ]
                output[y, x, c if c is not None else ...] = np.sum(roi * kernel)

    ## Convoluted image is shown
    st.header("🆚 Convoluted Image (After Processing)")
    st.image(output.astype(np.uint8))

    # Download Button
    _, img_encoded = cv2.imencode(".png", output.astype(np.uint8))
    img_byte_arr = img_encoded.tobytes()
        
    st.download_button(
            label="⬇️ Save Convoluted Image",
            data=img_byte_arr,
            file_name="image.png",
            mime="image/png"
            )
    
    return output.astype(np.uint8)
