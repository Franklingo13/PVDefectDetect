import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *

## Script para analizar las métricas obtenidas en el archivo CSV

path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
nombre_archivo = "metricsMMC.csv"
print("Directorio a leer:", path)

contrast_original = []  # Lista para almacenar los valores de contraste del dataset original
contrast_MMC = []  # Lista para almacenar los valores de contraste del dataset con MMC
cir_MMC =[]  # Lista para almacenar los valores de CIR del dataset con MMC
pl_MMC = []  # Lista para almacenar los valores de PL del dataset con MMC

# Leer el archivo CSV
df = pd.read_csv(os.path.join(path, nombre_archivo))