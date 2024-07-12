import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

# Aplicar diferentes correcciones a un dataset de imágenes

# Directorio de las imágenes
path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino_30W\Poli30W_V24_I2.5_t30\EL"  
bg_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino_30W\Poli30W_V24_I2.5_t30\Fondo"
out_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino_30W\Poli30W_V24_I2.5_t30"
salidas_path = read_folder_path(out_path)
dataset_path = read_folder_path(path)
BG_dataset_path = read_folder_path(bg_path)
print("Directorio imagenes: ", dataset_path)
print("Directorio fondo: ", BG_dataset_path)

dataset = read_images(dataset_path)
datasetBG = read_images(BG_dataset_path)
print("Número de imágenes en el dataset:", len(dataset))

# muestra la imagen datset[0]
plt.imshow(dataset[0], cmap='gray')
plt.show()

# Aplicar el algoritmo `SubtractBG(imageEL, imageBG)` al dataset
datasetNoBG = []
for i in range(len(dataset)):
    imageNoBG = SubtractBG(dataset[i], datasetBG[i])
    datasetNoBG.append(imageNoBG)
print("SusbstractBG aplicado")
plt.imshow(datasetNoBG[0], cmap='gray')
plt.show()

# Aplicar el algoritmo de eliminación de artefactos a las imágenes sin fondo
datasetNoBG_noartefacts = []
for i in range(len(datasetNoBG)):
    imageNoBG_noartefacts = correctArtefacts(datasetNoBG[i], 0.1)
    datasetNoBG_noartefacts.append(imageNoBG_noartefacts)
print("Artefactos corregidos")
plt.imshow(datasetNoBG_noartefacts[0],  cmap='gray')
plt.show()

# Aplicar el algoritmo de mejora de contraste CLAHE) a las imágenes sin fondo
datasetNoBG_noartefacts_CLAHE = []
for i in range(len(datasetNoBG_noartefacts)):
    imageNoBG_noartefacts_CLAHE = CLAHE(datasetNoBG_noartefacts[i])
    datasetNoBG_noartefacts_CLAHE.append(imageNoBG_noartefacts_CLAHE)
print("CLAHE aplicado")
plt.imshow(datasetNoBG_noartefacts_CLAHE[0], cmap='gray')
plt.show()

# Aplicar el algoritmo de get_max_min_mean a las imágenes 
mean_image, max_image, min_image = get_mean_max_min_image(datasetNoBG_noartefacts_CLAHE)
print("mean_image, max_image, min_image calculados")
plt.imshow(mean_image, cmap='gray')
plt.show()

## Creación de un dataset con las imágenes mejoradas, que se almacena en el directorio `salidas_path/nombre_carpeta`
nombre_carpeta = "Mono1_Cracked_ImagenesCorregidas"
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "mean_image.jpg"), mean_image)
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "max_image.jpg"), max_image)
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "min_image.jpg"), min_image)

