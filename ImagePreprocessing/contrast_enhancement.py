# Importación de librerías
import numpy as np
import cv2

# Fichero para almacenar los diferentes métodos de mejora de contraste


def MMC(image, n_iterations):
    """
    Función que implementa el algoritmo de Mejora de Contraste por Morfología
    de Contraste Máximo (MMCE)

    Parámetros:
    image: Imagen de entrada en escala de grises.
    n_iterations: Número de iteraciones a realizar.

    Retorna:
    enhanced_image: Imagen con aumento de contraste.
    """
    # Primera etapa del algoritmo: Obtener escalas de brillo y oscuridad
    scales = [cv2.getStructuringElement(cv2.MORPH_RECT, (2*i+1, 2*i+1)) for i in range(1, n_iterations+1)]
    WTH_scales = [cv2.subtract(image, cv2.morphologyEx(image, cv2.MORPH_TOPHAT, scale)) for scale in scales]
    BTH_scales = [cv2.subtract(cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, scale), image) for scale in scales]
    
    # Segunda etapa del algoritmo: Obtener sustracciones de escalas
    WTH_diff_scales = [cv2.subtract(WTH_scales[i], WTH_scales[i-1]) for i in range(1, len(WTH_scales))]
    BTH_diff_scales = [cv2.subtract(BTH_scales[i], BTH_scales[i-1]) for i in range(1, len(BTH_scales))]
    
    # Tercera etapa del algoritmo: Calcular máximos valores
    WTH_max_scale = max(WTH_scales, key=np.max)
    BTH_max_scale = max(BTH_scales, key=np.max)
    WTH_diff_max_scale = max(WTH_diff_scales, key=np.max)
    BTH_diff_max_scale = max(BTH_diff_scales, key=np.max)
    
    # Etapa final: Obtener la imagen con aumento de contraste
    enhanced_image = cv2.add(cv2.subtract(cv2.add(image, WTH_max_scale), BTH_max_scale), cv2.subtract(WTH_diff_max_scale, BTH_diff_max_scale))
    
    return enhanced_image

def CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Función que implementa el algoritmo de Mejora de Contraste Adaptativo
    basado en Histogramas Limitados (CLAHE)

    Parámetros:
    image: Imagen de entrada en escala de grises.
    clip_limit: Límite de recorte para el histograma.
    tile_grid_size: Tamaño de la cuadrícula para la ecualización.

    Retorna:
    enhanced_image: Imagen con aumento de contraste.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)
    
    return enhanced_image

def HE(image):
    """
    Función que implementa el algoritmo de Ecualización de Histograma (HE)

    Parámetros:
    image: Imagen de entrada en escala de grises.

    Retorna:
    enhanced_image: Imagen con aumento de contraste.
    """
    enhanced_image = cv2.equalizeHist(image)
    
    return enhanced_image

# Eliminar pixeles calientes: radio de pixel = 2.0; Threshold = 50

def remove_hot_pixels(image):
    """
    Elimina los pixeles calientes de una imagen EL.

    Parámetros:
        image: La imagen EL de entrada.
        radius: Radio del pixel caliente.
        threshold: Umbral para la eliminación de pixeles calientes.

    Devuelve:
        La imagen sin pixeles calientes.
    """
    radius=2.0
    threshold=180
    ret, mask = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    image = cv2.inpaint(image, mask, 2, cv2.INPAINT_TELEA)

    return image

def SubtractBG(imageEL, inageBG):
    """
    Resta una imagen de fondo a una imagen de EL.

    Parámetros:
        imageEL: Imagen de EL.
        imageBG: Imagen de fondo.

    Devuelve:
        La imagen EL con la imagen de fondo restada.
    """
    imageEL = cv2.subtract(imageEL, inageBG)

    return imageEL