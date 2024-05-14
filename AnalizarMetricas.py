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

# Obtener los valores de contraste, PL y CIR
contrast_original = df["Contraste EL"].to_list()
contrast_MMC = df["Contraste MMC"].to_list()
cir_MMC = df["CIR MMC"].to_list()
pl_MMC = df["PL MMC"].to_list()


## Graficar los valores de contraste `contrast_original` y `contrast_MMC` con sus intervalos de confianza y medias

# Calcular la media y el intervalo de confianza para los valores de contraste
mean_original = np.mean(contrast_original)
mean_MMC = np.mean(contrast_MMC)
conf_interval_original = np.percentile(contrast_original, [2.5, 97.5])
conf_interval_MMC = np.percentile(contrast_MMC, [2.5, 97.5])

# Gráfica de contraste
plt.figure()
plt.plot(contrast_original, label="Contraste EL")
plt.plot(contrast_MMC, label="Contraste MMC")
#plt.axhline(mean_original, color="blue", linestyle="--", label="Media EL")
#plt.axhline(mean_MMC, color="orange", linestyle="--", label="Media MMC")
plt.fill_between(range(len(contrast_original)), conf_interval_original[0], conf_interval_original[1], color="blue", alpha=0.2, label="Intervalo de confianza EL")
plt.fill_between(range(len(contrast_MMC)), conf_interval_MMC[0], conf_interval_MMC[1], color="orange", alpha=0.2, label="Intervalo de confianza MMC")
plt.title("Contraste")
plt.xlabel("Imagen")
plt.ylabel("Contraste")
plt.legend()
plt.show()

# # Gráfica de CIR
# plt.figure()
# plt.plot(cir_MMC, label="CIR MMC")
# plt.title("CIR")
# plt.xlabel("Imagen")
# plt.ylabel("CIR")
# plt.legend()
# plt.show()

# # Gráfica de PL
# plt.figure()
# plt.plot(pl_MMC, label="PL MMC")
# plt.title("PL")
# plt.xlabel("Imagen")
# plt.ylabel("PL")
# plt.legend()
# plt.show()