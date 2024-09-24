# Fichero para almacenar funciones para el manejo de archivos

# Importación de librerías
import os
from os import listdir, path
import numpy as np
import cv2
import matplotlib.patches as patches
from matplotlib import pyplot as plt

def read_images(directory, allow_color_images=False):
    """
    Lee todas las imágenes en un directorio.

    Args:
        directory (str): Directorio con las imágenes.
        allow_color_images (bool): Permite el uso de imágenes a color.

    Returns:
        list: Lista con las imágenes.
    """
    # Leer todas las imágenes en el directorio
    images = []
    for f in listdir(directory):
        if f.endswith('.jpg') or f.endswith('.png'):
            image = cv2.imread(path.join(directory, f), cv2.IMREAD_GRAYSCALE)
            if allow_color_images:
                image = cv2.imread(path.join(directory, f), cv2.IMREAD_COLOR)
            images.append(image)
    return images

def read_folder_path(folder_path):
    """
    Función que lee una ruta de una carpeta y la reescribe de forma que sea legible para el sistema operativo.

    Args:
        folder_path (str): Ruta de la carpeta.

    Returns:
        str: Ruta de la carpeta legible para el sistema operativo.
    """
    folder_path = folder_path.replace("\\", "/")
    if folder_path[-1] != "/":
        folder_path += "/"
    return folder_path

def mostrar_imagen_con_roi(imagen, roi, titulo, esquina_superior_izquierda, ancho, alto):
    """
    Muestra una imagen con una zona ampliada o ROI.

    Args:
        imagen (numpy.ndarray): La imagen a mostrar.
        roi (numpy.ndarray): La zona ampliada (ROI).
        titulo (str): El título de la ventana.
        esquina_superior_izquierda (tuple): Una tupla (x, y) que representa las coordenadas de la esquina superior izquierda de la ROI.
        ancho (int): El ancho de la ROI.
        alto (int): El alto de la ROI.
    """
    # Crear una figura de Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Mostrar la imagen original en el primer eje
    axes[0].imshow(imagen, cmap='gray')
    axes[0].set_title(titulo)
    axes[0].axis('off')

    # Dibujar un rectángulo alrededor de la ROI en la imagen original
    rect = patches.Rectangle(esquina_superior_izquierda, ancho, alto, linewidth=1, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    # Mostrar la zona ampliada (ROI) en el segundo eje
    axes[1].imshow(roi, cmap='gray')
    axes[1].set_title('Zona ampliada')
    axes[1].axis('off')

    # Ajustar el layout de la figura
    plt.tight_layout()

    # Mostrar la figura
    plt.show()

def save_image(image, directory, filename):
    """
    Guarda una imagen en un directorio.

    Args:
        image (numpy.ndarray): Imagen a guardar.
        directory (str): Directorio donde se guardará la imagen.
        filename (str): Nombre de la imagen.
    """
    # Verificar si el directorio existe
    if not path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path.join(directory, filename), image)