# Importar librerías
import os
import numpy as np
import cv2

# Fichero para almacenar los diferentes métodos de evaluación de algoritmos

# Función para medir la métrica Contrast en imágenes en escala de grises
def contrast_metric(image):
    """
    Función que calcula el contraste de una imagen en escala de grises.

    Parámetros:
    image: Imagen en escala de grises.

    Retorna:
    C: Valor del contraste de la imagen.
    """
    # Calcular la intensidad media de la imagen
    E = np.mean(image)

    # Obtener el número de niveles de grises (L)
    num_gray_levels = np.max(image) + 1

    # Calcular la probabilidad de ocurrencia de cada valor de píxel
    hist, _ = np.histogram(image, bins=num_gray_levels, range=(0, num_gray_levels))
    P = hist / float(np.sum(hist))

    # Calcular el contraste
    C = np.sqrt(np.sum([(j - E)**2 * P[j] for j in range(num_gray_levels)]))

    return C

# Función para medir la métrica contrast Improvement ratio (CIR) en imágenes en escala de grises
def CIR(original_image, enhanced_image):
    """
    Función que calcula el índice de mejora de contraste (CIR) entre dos imágenes en escala de grises.

    Parámetros:
    original_image: Imagen original en escala de grises.
    enhanced_image: Imagen mejorada en escala de grises.

    Retorna:
    CIR: Valor del índice de mejora de contraste.
    """
    # Calcular el tamaño de la imagen
    M, N = original_image.shape

    # Inicializar las sumas para el numerador y el denominador
    numerator_sum = 0
    denominator_sum = 0

    # Calcular el CIR
    for u in range(1, M-1):
        for v in range(1, N-1):
            # Calcular el contraste local para la imagen original
            w_original = abs(original_image[u, v] - np.mean(original_image[u-1:u+2, v-1:v+2])) / (original_image[u, v] + np.mean(original_image[u-1:u+2, v-1:v+2]))
            
            # Calcular el contraste local para la imagen mejorada
            w_enhanced = abs(enhanced_image[u, v] - np.mean(enhanced_image[u-1:u+2, v-1:v+2])) / (enhanced_image[u, v] + np.mean(enhanced_image[u-1:u+2, v-1:v+2]))

            # Actualizar las sumas para el numerador y el denominador
            numerator_sum += (w_original - w_enhanced)**2
            denominator_sum += w_original**2

    # Calcular el CIR
    CIR = numerator_sum / denominator_sum

    return CIR

# Función para calcular el PSNR entre las imágenes original y mejorada
def PSNR(original_image, enhanced_image):
    """
    Función que calcula el Pico de la Relación de Señal a Ruido (PSNR) entre dos imágenes en escala de grises.

    Parámetros:
    original_image: Imagen original en escala de grises.
    enhanced_image: Imagen mejorada en escala de grises.

    Retorna:
    psnr: Valor del PSNR entre las dos imágenes.
    """
    # Calcular el error cuadrático medio
    mse = np.mean((original_image - enhanced_image)**2)

    # Calcular el PSNR
    psnr = 10 * np.log10((255**2) / mse)

    return psnr

# Función para calcular el ínidice de borrosidad en imágenes en escala de grises
def blur_metric(enhanced_image):
    """
    Función que calcula el índice de borrosidad de una imagen en escala de grises.

    Parámetros:
    enhanced_image: Imagen mejorada en escala de grises.

    Retorna:
    blur_index: Valor del índice de borrosidad de la imagen.
    """
    # Calcular el índice de borrosidad
    M, N = enhanced_image.shape
    p_xy = np.sin(np.pi / 2 * (1 - enhanced_image / np.max(enhanced_image)))
    blur_index = (2 / (M * N)) * np.sum(np.minimum(p_xy, (1 - p_xy)))
    return blur_index

# Función para calcular PL metric en imágenes en escala de grises
def PL(original_image, enhanced_image):
    """
    Función que calcula el índice de calidad de una imagen en escala de grises.

    Parámetros:
    original_image: Imagen original en escala de grises.
    enhanced_image: Imagen mejorada en escala de grises.

    Retorna:
    PL: Valor del índice de calidad de la imagen.
    """
    psnr = PSNR(original_image, enhanced_image)
    blur = blur_metric(enhanced_image)

    # Calcular PL
    PL = psnr / blur
    return PL
