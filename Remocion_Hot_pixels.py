import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *

# Aplicación de la remoción de pixeles calientes a un dataset, y guardado de las imágenes mejoradas

path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino\Poli_Sup_V44.5_I9.16_t30\Poli_Sup_V40_I4.5_t28\Poli_Sup_V40_I4.5_t28_NoBG_CLAHE_mean_max_min"
salidas_path = read_folder_path(
    r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino\Poli_Sup_V44.5_I9.16_t30\Poli_Sup_V40_I4.5_t28")
dataset_path = read_folder_path(path)
print("Directorio a leer:", dataset_path)

dataset = read_images(dataset_path)
print("Número de imágenes en el dataset:", len(dataset))

# Aplicar el algoritmo `remove_hot_pixels(image)` al dataset
dataset_no_hot_pixels = []
for i, image in enumerate(dataset):
    image_no_hot_pixels = correctArtefacts(image, 0.1)
    dataset_no_hot_pixels.append(image_no_hot_pixels)

## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/nombre_carpeta`
nombre_carpeta = "Poli_Sup_V40_I4.5_t28_NoBG_CLAHE_mean_max_min_no_hot_pixels"
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
for i, image in enumerate(dataset_no_hot_pixels):
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "imagen"+str(i)+".jpg"), image)
print("Imágenes guardadas en ", nombre_carpeta)
print("Número de imágenes: ", len(dataset_no_hot_pixels))

