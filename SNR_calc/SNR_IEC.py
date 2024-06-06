import numpy as np
import matplotlib.pyplot as plt

# Función para el cálculo de la relación señal a ruido (SNR) promedio SNR50

def SNR_IEC(i1, i2, ibg, allow_color_images=False):
    """
    Calculla la relación señal a ruido (SNR) promedio SNR50 según la norma IEC 60904-13

    Parámetros

    i1: numpy.ndarray
        Imagen de referencia
    i2: numpy.ndarray
        Imagen a comparar
    ibg: float
        Imagen del fondo
    allow_color_images: bool
        Permite el uso de imágenes a color

    Retorna

    snr50: float
        Relación señal a ruido (SNR) promedio SNR50
    """
    # Validar que las imágenes sean del tipo float64 (doble precisión):
    i1 = i1.astype(np.float64)
    i2 = i2.astype(np.float64)
    #i1 = np.asfarray(i1)
    #i2 = np.asfarray(i2)
    if np.any(ibg != 0):
        ibg = ibg.astype(np.float64)
        #ibg = np.asfarray(ibg)
        assert i1.shape == ibg.shape, "Las imágenes deben tener la misma resolución"
    # Validar que las imágenes tengan la misma resolución:
    if i1.shape != i2.shape:
        raise ValueError("Las imágenes deben tener la misma resolución")
    
    # Validar que las imágenes sean en escala de grises:
    if not allow_color_images:
        if len(i1.shape) == 3:
            raise ValueError("Las imágenes deben ser en escala de grises")
        
    # SNR definido por la norma IEC 60904-13
    signal = 0.5 * (i1 + i2) -ibg
    noise = 0.5**0.5 * np.abs(i1 - i2) * ((2 / np.pi)**0.5)
    signal = signal.sum()
    noise = noise.sum()

    snr50 = signal / noise

    return snr50
    