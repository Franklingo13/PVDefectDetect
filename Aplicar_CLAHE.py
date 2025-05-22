import os
import cv2
import matplotlib.pyplot as plt
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
from EvaluationMetrics.evaluationMetrics import *

## Aplicación del algoritmo CLAHE a una sola imagen,  y guardado de las imágenes mejoradas

# Ruta de la imagen a procesar
image_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\ImagenesTesis\Articulo\preprocessing_results_Poli\Celdas\poli2_transformed.jpg" 
salidas_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\ImagenesTesis\Articulo\preprocessing_results_Poli\Celdas"

# Leer la imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar CLAHE
image_CLAHE = CLAHE(image)

# Guardar la imagen procesada
os.makedirs(os.path.join(salidas_path), exist_ok=True)
output_path = os.path.join(salidas_path, "imagen_transformed_CLAHE.jpg")
cv2.imwrite(output_path, image_CLAHE)

print(f"Imagen procesada y guardada en: {output_path}")

