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
