import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicación del algoritmo CLAHE a un dataset,  y guardado de las imágenes mejoradas
salidas_path = read_folder_path(
    r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\panel_260W")
dataset_path = read_folder_path(
    r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\panel_260W\V44.5_I9.16_t_NoBG")

dataset = read_images(dataset_path)

dataset_CLAHE = []

for i, image in enumerate(dataset):
    dataset_CLAHE.append(CLAHE(image))


## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/nombre_carpeta`
nombre_carpeta = "V44.5_I9.16_t_NoBG_CLAHE"
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
for i, image in enumerate(dataset_CLAHE):
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "imagen"+str(i)+".jpg"), image)
print("Imágenes guardadas en: ", nombre_carpeta)
print("Número de imágenes: ", len(dataset_CLAHE))

