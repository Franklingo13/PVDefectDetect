import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

original_path = r"E:/Panel_30W/P13/V24.2_I2.50_t30/JPEG_8"
path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste\datasetNoBG_MMC_mean_max_min"  
salidas_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
dataset_path = read_folder_path(path)
nombre_archivo = "metrics_datasetNoBG_CLAHE.csv"
print("Directorio a leer:", dataset_path)
print("Directorio original:", original_path)

datasetOriginal = read_images(original_path)
dataset = read_images(dataset_path)
print("Número de imágenes en el dataset original: ", len(datasetOriginal))
print("Número de imágenes en el dataset:", len(dataset))


# Condicional para evaluar que el dataset tenga más de 3 imágenes
# Si tiene solo tres imágenes, significa que es un dataset con Mean, Max y Min
if len(dataset) > 3:
    ## Obtención de métricas `contrast_metric`, `PL` y `CIR` para las imágenes del dataset
    contrast_values = []
    for image in dataset:
        contrast_values.append(contrast_metric(image))
    print("Contraste promedio de las imágenes:", np.mean(contrast_values))

    PL_values = []
    for imageMejorada, imageOriginal in zip(dataset, datasetOriginal):
        PL_values.append(PL(imageOriginal, imageMejorada))
    print("PL promedio de las imágenes:", np.mean(PL_values))

    CIR_values = []
    for i, (imageMejorada, imageOriginal) in enumerate(zip(dataset, datasetOriginal)):
        if i % 10 == 0:
            print("Imagen: ", i)
        CIR_values.append(CIR(imageOriginal, imageMejorada))
    print("CIR promedio de las imágenes:", np.mean(CIR_values))

    ## Anotación de los resultados en un archivo CSV
    df = pd.DataFrame({
        "Contraste": contrast_values,
        "PL": PL_values,
        "CIR": CIR_values
    })
    df.to_csv(os.path.join(salidas_path, nombre_archivo), index=False)
    print("Métricas guardadas en ", nombre_archivo)

else:
    print("El dataset tiene menos de 3 imágenes, no se puede calcular PL y CIR")