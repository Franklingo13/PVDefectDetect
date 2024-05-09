import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicación del algoritmo `get_mean_max_min_image(dataset)` a un dataset,  y guardado de las imágenes mejoradas
# en este caso: datasetNoBG_CLAHE
path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste\datasetNoBG_CLAHE"  
salidas_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
dataset_path = read_folder_path(path)
print("Directorio a leer:", dataset_path)

dataset = read_images(dataset_path)
print("Número de imágenes en el dataset:", len(dataset))

# Aplicar el algoritmo `get_mean_max_min_image(dataset)` al dataset
mean_image, max_image, min_image = get_mean_max_min_image(dataset)

## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/datasetNoBG_CLAHE_mean_max_min`
os.makedirs(os.path.join(salidas_path, "datasetNoBG_CLAHE_mean_max_min"), exist_ok=True)
cv2.imwrite(os.path.join(salidas_path, "datasetNoBG_CLAHE_mean_max_min", "mean_image.jpg"), mean_image)
cv2.imwrite(os.path.join(salidas_path, "datasetNoBG_CLAHE_mean_max_min", "max_image.jpg"), max_image)
cv2.imwrite(os.path.join(salidas_path, "datasetNoBG_CLAHE_mean_max_min", "min_image.jpg"), min_image)
print("Imágenes guardadas en datasetNoBG_CLAHE_mean_max_min")