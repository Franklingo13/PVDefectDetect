import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

salidas_path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
datasetEL = read_images("E:/Panel_30W/P13/V24.2_I2.50_t30/JPEG_8")
datasetMMC_path = salidas_path + "datasetMMC"
datasetMMC = read_images(datasetMMC_path)
datasetCLAHE_path = salidas_path + "datasetCLAHE"
datasetCLAHE = read_images(datasetCLAHE_path)
datasetHE_path = salidas_path + "datasetHE"
datasetHE = read_images(datasetHE_path)
datasetNoBG_path = salidas_path + "datasetNoBG"
datasetNoBG = read_images(datasetNoBG_path)

# Imprime el número de imágenes en cada dataset
print("Número de imágenes en datasetEL:", len(datasetEL))
print("Número de imágenes en datasetMMC:", len(datasetMMC))
print("Número de imágenes en datasetCLAHE:", len(datasetCLAHE))
print("Número de imágenes en datasetHE:", len(datasetHE))
print("Número de imágenes en datasetNoBG:", len(datasetNoBG))

## Aplicación del algoritmo get_mean_max_min_image(datasetMMC):
mean_imageMMC, max_imageMMC, min_imageMMC = get_mean_max_min_image(datasetMMC)
mean_imageCLAHE, max_imageCLAHE, min_imageCLAHE = get_mean_max_min_image(datasetCLAHE)
mean_imageHE, max_imageHE, min_imageHE = get_mean_max_min_image(datasetHE)
mean_imageNoBG, max_imageNoBG, min_imageNoBG = get_mean_max_min_image(datasetNoBG)

## Creación de un dataset con las imágenes mejoradas de MMC
os.makedirs(salidas_path + "datasetMMC_MeanMaxMin", exist_ok=True)
cv2.imwrite(os.path.join(salidas_path + "datasetMMC_MeanMaxMin", "mean_imageMMC.jpg"), mean_imageMMC)
cv2.imwrite(os.path.join(salidas_path + "datasetMMC_MeanMaxMin", "max_imageMMC.jpg"), max_imageMMC)
cv2.imwrite(os.path.join(salidas_path + "datasetMMC_MeanMaxMin", "min_imageMMC.jpg"), min_imageMMC)
print("Imágenes guardadas en datasetMMC_MeanMaxMin")

## Creación de un dataset con las imágenes mejoradas de CLAHE
os.makedirs(salidas_path + "datasetCLAHE_MeanMaxMin", exist_ok=True)
cv2.imwrite(os.path.join(salidas_path + "datasetCLAHE_MeanMaxMin", "mean_imageCLAHE.jpg"), mean_imageCLAHE)
cv2.imwrite(os.path.join(salidas_path + "datasetCLAHE_MeanMaxMin", "max_imageCLAHE.jpg"), max_imageCLAHE)
cv2.imwrite(os.path.join(salidas_path + "datasetCLAHE_MeanMaxMin", "min_imageCLAHE.jpg"), min_imageCLAHE)
print("Imágenes guardadas en datasetCLAHE_MeanMaxMin")

## Creación de un dataset con las imágenes mejoradas de HE
os.makedirs(salidas_path + "datasetHE_MeanMaxMin", exist_ok=True)
cv2.imwrite(os.path.join(salidas_path + "datasetHE_MeanMaxMin", "mean_imageHE.jpg"), mean_imageHE)
cv2.imwrite(os.path.join(salidas_path + "datasetHE_MeanMaxMin", "max_imageHE.jpg"), max_imageHE)
cv2.imwrite(os.path.join(salidas_path + "datasetHE_MeanMaxMin", "min_imageHE.jpg"), min_imageHE)
print("Imágenes guardadas en datasetHE_MeanMaxMin")

## Creación de un dataset con las imágenes mejoradas de NoBG
os.makedirs(salidas_path + "datasetNoBG_MeanMaxMin", exist_ok=True)
cv2.imwrite(os.path.join(salidas_path + "datasetNoBG_MeanMaxMin", "mean_imageNoBG.jpg"), mean_imageNoBG)
cv2.imwrite(os.path.join(salidas_path + "datasetNoBG_MeanMaxMin", "max_imageNoBG.jpg"), max_imageNoBG)
cv2.imwrite(os.path.join(salidas_path + "datasetNoBG_MeanMaxMin", "min_imageNoBG.jpg"), min_imageNoBG)
print("Imágenes guardadas en datasetNoBG_MeanMaxMin")

