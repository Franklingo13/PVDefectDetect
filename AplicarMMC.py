import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicar el algoritmo `MMC(image, n_iterations)` a un dataset,  y guardado de las imágenes mejoradas
path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste\datasetNoBG"  
salidas_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
dataset_path = read_folder_path(path)
print("Directorio a leer:", dataset_path)

dataset = read_images(dataset_path)
print("Número de imágenes en el dataset:", len(dataset))

# Aplicar el algoritmo `MMC(image, n_iterations)` al dataset
n_iterations = 7
dataset_MMC = []
for i, image in enumerate(dataset):
    dataset_MMC.append(MMC(image, n_iterations))

## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/datasetNoBG_MMC`
os.makedirs(os.path.join(salidas_path, "datasetNoBG_MMC"), exist_ok=True)
for i, image in enumerate(dataset_MMC):
    cv2.imwrite(os.path.join(salidas_path, "datasetNoBG_MMC", f"NoBGMMC_{i}.jpg"), image)
print("Imágenes guardadas en datasetNoBG_MMC")