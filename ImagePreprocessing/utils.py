# Fichero para almacenar funciones para el manejo de archivos

# Importación de librerías
import os
from os import listdir, path
import numpy as np
import cv2

# Función para la lectura de imágenes
def read_images(directory, allow_color_images=False):
    """
    Lee todas las imágenes en un directorio

    Parámetros

    directory: str
        Directorio con las imágenes
    allow_color_images: bool
        Permite el uso de imágenes a color

    Retorna

    images: list
        Lista con las imágenes
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

# Función para guardar una imagen en un directorio
def save_image(image, directory, filename):
    """
    Guarda una imagen en un directorio

    Parámetros

    image: np.array
        Imagen a guardar
    directory: str
        Directorio donde se guardará la imagen
    filename: str
        Nombre de la imagen
    """
    # Verificar si el directorio existe
    if not path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path.join(directory, filename), image)

"""# Almacena el dataset en el output_noBG_path con nombres numerados
        filename = f"EL_noBG_{i}.jpg"
        output_filepath = os.path.join(output_noBG_path, filename)

        # Convierte la matriz de imagen a tipo uint8 antes de guardarla
        dataset_noBG_uint8 = cv2.normalize(dataset_noBG[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        cv2.imwrite(output_filepath, dataset_noBG_uint8)
        print(f"Imagen {i} guardada como {filename}")
        
        """