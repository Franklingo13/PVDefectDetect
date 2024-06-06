import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicación del algoritmo subsctractBG a un dataset,  y guardado de las imágenes mejoradas

path = r"E:\Panel_260W\V40_I4.5_t\JPEG"  
bg_path = r"E:\Panel_260W\V40_I4.5_t\fondo_jpeg"
salidas_path = read_folder_path(r"E:\Panel_260W")
dataset_path = read_folder_path(path)
BG_dataset_path = read_folder_path(bg_path)
print("Directorio imagenes: ", dataset_path)
print("Directorio fondo: ", BG_dataset_path)

dataset = read_images(dataset_path)
datasetBG = read_images(BG_dataset_path)
print("Número de imágenes en el dataset:", len(dataset))

# Aplicar el algoritmo `SubtractBG(imageEL, imageBG)` al dataset
datasetNoBG = []
for i in range(len(dataset)):
    imageNoBG = SubtractBG(dataset[i], datasetBG[i])
    datasetNoBG.append(imageNoBG)

## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/nombre_carpeta`
nombre_carpeta = "V40_I4.5_t_NoBG"
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
for i in range(len(datasetNoBG)):
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "imagen"+str(i)+".jpg"), datasetNoBG[i])
print("Imágenes guardadas en ", nombre_carpeta)

# Aplicar el algoritmo CLAHE al dataset sin fondo
dataset_CLAHE = []
for i, image in enumerate(datasetNoBG):
    dataset_CLAHE.append(CLAHE(image))

nombre_carpeta = "V40_I4.5_t_NoBG_CLAHE"
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
for i, image in enumerate(dataset_CLAHE):
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "imagen"+str(i)+".jpg"), image)
print("Imágenes guardadas en ", nombre_carpeta)
print("Número de imágenes: ", len(dataset_CLAHE))
