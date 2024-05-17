import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicación del algoritmo CLAHE a un dataset,  y guardado de las imágenes mejoradas
salidas_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
dataset_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste\datasetNoBG")

dataset = read_images(dataset_path)

dataset_CLAHE = []

for i, image in enumerate(dataset):
    dataset_CLAHE.append(CLAHE(image))


## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/datasetCLAHE`
os.makedirs(os.path.join(salidas_path, "datasetNoBG_CLAHE"), exist_ok=True)
for i, image in enumerate(dataset_CLAHE):
    cv2.imwrite(os.path.join(salidas_path, "datasetNoBG_CLAHE", f"NoBGCLAHE_{i}.jpg"), image)
print("Imágenes guardadas en datasetNoBG_CLAHE")
print("Número de imágenes en datasetNoBG_CLAHE:", len(dataset_CLAHE))

