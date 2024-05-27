import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from ImagePreprocessing.utils import *

## Script para analizar las métricas obtenidas en el archivo CSV

path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
archivo_MMCE = "metricsMMC.csv"
archivo_CLAHE ="metricsCLAHE.csv"
archivo_HE = "metricsHE.csv"
acrhivo_NoBG = "metricsNoBG.csv"
archivo_NoBG_CLAHE = "metrics_datasetNoBG_CLAHE.csv"
archivo_NoBG_MMCE = "metrics_datasetNoBG_MMC.csv"
archivo_img_promediadas = "metricsMeanMaxMin_datasetEL.csv"



contrast_original = []  # Lista para almacenar los valores de contraste del dataset original
contrast_MMCE = []  # Lista para almacenar los valores de contraste del dataset con MMC
cir_MMCE =[]  # Lista para almacenar los valores de CIR del dataset con MMC
pl_MMCE = []  # Lista para almacenar los valores de PL del dataset con MMC+
contrast_CLAHE = []  # Lista para almacenar los valores de contraste del dataset con CLAHE
cir_CLAHE = []  # Lista para almacenar los valores de CIR del dataset con CLAHE
pl_CLAHE = []  # Lista para almacenar los valores de PL del dataset con CLAHE
contrast_HE = []  # Lista para almacenar los valores de contraste del dataset con HE
cir_HE = []  # Lista para almacenar los valores de CIR del dataset con HE
pl_HE = []  # Lista para almacenar los valores de PL del dataset con HE
contrast_NoBG = []  # Lista para almacenar los valores de contraste del dataset sin fondo
cir_NoBG = []  # Lista para almacenar los valores de CIR del dataset sin fondo
pl_NoBG = []  # Lista para almacenar los valores de PL del dataset sin fondo
contrast_NoBG_CLAHE = []  # Lista para almacenar los valores de contraste del dataset sin fondo con CLAHE
cir_NoBG_CLAHE = []  # Lista para almacenar los valores de CIR del dataset sin fondo con CLAHE
pl_NoBG_CLAHE = []  # Lista para almacenar los valores de PL del dataset sin fondo con CLAHE
contrast_NoBG_MMCE = []  # Lista para almacenar los valores de contraste del dataset sin fondo con MMC
cir_NoBG_MMCE = []  # Lista para almacenar los valores de CIR del dataset sin fondo con MMC
pl_NoBG_MMCE = []  # Lista para almacenar los valores de PL del dataset sin fondo con MMC
# contrast_img_promediadas = []  # Lista para almacenar los valores de contraste de las imágenes promediadas
# cir_img_promediadas = []  # Lista para almacenar los valores de CIR de las imágenes promediadas
# pl_img_promediadas = []  # Lista para almacenar los valores de PL de las imágenes promediadas

# Leer los archivos CSV
df = pd.read_csv(os.path.join(path, archivo_MMCE))
df_CLAHE = pd.read_csv(os.path.join(path, archivo_CLAHE))
df_HE = pd.read_csv(os.path.join(path, archivo_HE))
df_NoBG = pd.read_csv(os.path.join(path, acrhivo_NoBG))
df_NoBG_CLAHE = pd.read_csv(os.path.join(path, archivo_NoBG_CLAHE))
df_NoBG_MMCE = pd.read_csv(os.path.join(path, archivo_NoBG_MMCE))

# Obtener los valores de contraste, PL y CIR del dataset con MMC
contrast_original = df["Contraste EL"].to_list()
contrast_MMCE = df["Contraste MMC"].to_list()
cir_MMCE = df["CIR MMC"].to_list()
pl_MMCE = df["PL MMC"].to_list()
# Obtener los valores de contraste, PL y CIR del dataset con CLAHE
contrast_CLAHE = df_CLAHE["Contraste CLAHE"].to_list()
cir_CLAHE = df_CLAHE["CIR CLAHE"].to_list()
pl_CLAHE = df_CLAHE["PL CLAHE"].to_list()
# Obtener los valores de contraste, PL y CIR del dataset con HE
contrast_HE = df_HE["Contraste HE"].to_list()
cir_HE = df_HE["CIR HE"].to_list()
pl_HE = df_HE["PL HE"].to_list()
# Obtener los valores de contraste, PL y CIR del dataset sin fondo
contrast_NoBG = df_NoBG["Contraste NoBG"].to_list()
cir_NoBG = df_NoBG["CIR NoBG"].to_list()
pl_NoBG = df_NoBG["PL NoBG"].to_list()
# Obtener los valores de contraste, PL y CIR del dataset sin fondo con CLAHE
contrast_NoBG_CLAHE = df_NoBG_CLAHE["Contraste"].to_list()
cir_NoBG_CLAHE = df_NoBG_CLAHE["CIR"].to_list()
pl_NoBG_CLAHE = df_NoBG_CLAHE["PL"].to_list()
# Obtener los valores de contraste, PL y CIR del dataset sin fondo con MMC
contrast_NoBG_MMCE = df_NoBG_MMCE["Contraste"].to_list()
cir_NoBG_MMCE = df_NoBG_MMCE["CIR"].to_list()
pl_NoBG_MMCE = df_NoBG_MMCE["PL"].to_list()



# Cálculo de las medias de los valores de contraste, PL y CIR
mean_contrast_original = np.mean(contrast_original)
mean_contrast_MMCE = np.mean(contrast_MMCE)
mean_cir_MMCE = np.mean(cir_MMCE)
mean_pl_MMCE = np.mean(pl_MMCE)
mean_contrast_CLAHE = np.mean(contrast_CLAHE)
mean_cir_CLAHE = np.mean(cir_CLAHE)
mean_pl_CLAHE = np.mean(pl_CLAHE)
mean_contrast_HE = np.mean(contrast_HE)
mean_cir_HE = np.mean(cir_HE)
mean_pl_HE = np.mean(pl_HE)
mean_contrast_NoBG = np.mean(contrast_NoBG)
mean_cir_NoBG = np.mean(cir_NoBG)
mean_pl_NoBG = np.mean(pl_NoBG)
mean_contrast_NoBG_CLAHE = np.mean(contrast_NoBG_CLAHE)
mean_cir_NoBG_CLAHE = np.mean(cir_NoBG_CLAHE)
mean_pl_NoBG_CLAHE = np.mean(pl_NoBG_CLAHE)
mean_contrast_NoBG_MMCE = np.mean(contrast_NoBG_MMCE)
mean_cir_NoBG_MMCE = np.mean(cir_NoBG_MMCE)
mean_pl_NoBG_MMCE = np.mean(pl_NoBG_MMCE)
# mean_contrast_img_promediadas = np.mean(contrast_img_promediadas)

# Cálculo de la desviación estándar de los valores de contraste, PL y CIR
std_contrast_original = np.std(contrast_original)
std_contrast_MMCE = np.std(contrast_MMCE)
std_cir_MMCE = np.std(cir_MMCE)
std_pl_MMCE = np.std(pl_MMCE)
std_contrast_CLAHE = np.std(contrast_CLAHE)
std_cir_CLAHE = np.std(cir_CLAHE)
std_pl_CLAHE = np.std(pl_CLAHE)
std_contrast_HE = np.std(contrast_HE)
std_cir_HE = np.std(cir_HE)
std_pl_HE = np.std(pl_HE)
std_contrast_NoBG = np.std(contrast_NoBG)
std_cir_NoBG = np.std(cir_NoBG)
std_pl_NoBG = np.std(pl_NoBG)
std_contrast_NoBG_CLAHE = np.std(contrast_NoBG_CLAHE)
std_cir_NoBG_CLAHE = np.std(cir_NoBG_CLAHE)
std_pl_NoBG_CLAHE = np.std(pl_NoBG_CLAHE)
std_contrast_NoBG_MMCE = np.std(contrast_NoBG_MMCE)
std_cir_NoBG_MMCE = np.std(cir_NoBG_MMCE)
std_pl_NoBG_MMCE = np.std(pl_NoBG_MMCE)
# std_contrast_img_promediadas = np.std(contrast_img_promediadas)

# Cálculo del error estándar de los valores de contraste, PL y CIR
sem_contrast_original = std_contrast_original / np.sqrt(len(contrast_original))
sem_contrast_MMCE = std_contrast_MMCE / np.sqrt(len(contrast_MMCE))
sem_cir_MMCE = std_cir_MMCE / np.sqrt(len(cir_MMCE))
sem_pl_MMCE = std_pl_MMCE / np.sqrt(len(pl_MMCE))
sem_contrast_CLAHE = std_contrast_CLAHE / np.sqrt(len(contrast_CLAHE))
sem_cir_CLAHE = std_cir_CLAHE / np.sqrt(len(cir_CLAHE))
sem_pl_CLAHE = std_pl_CLAHE / np.sqrt(len(pl_CLAHE))
sem_contrast_HE = std_contrast_HE / np.sqrt(len(contrast_HE))
sem_cir_HE = std_cir_HE / np.sqrt(len(cir_HE))
sem_pl_HE = std_pl_HE / np.sqrt(len(pl_HE))
sem_contrast_NoBG = std_contrast_NoBG / np.sqrt(len(contrast_NoBG))
sem_cir_NoBG = std_cir_NoBG / np.sqrt(len(cir_NoBG))
sem_pl_NoBG = std_pl_NoBG / np.sqrt(len(pl_NoBG))
sem_contrast_NoBG_CLAHE = std_contrast_NoBG_CLAHE / np.sqrt(len(contrast_NoBG_CLAHE))
sem_cir_NoBG_CLAHE = std_cir_NoBG_CLAHE / np.sqrt(len(cir_NoBG_CLAHE))
sem_pl_NoBG_CLAHE = std_pl_NoBG_CLAHE / np.sqrt(len(pl_NoBG_CLAHE))
sem_contrast_NoBG_MMCE = std_contrast_NoBG_MMCE / np.sqrt(len(contrast_NoBG_MMCE))
sem_cir_NoBG_MMCE = std_cir_NoBG_MMCE / np.sqrt(len(cir_NoBG_MMCE))
sem_pl_NoBG_MMCE = std_pl_NoBG_MMCE / np.sqrt(len(pl_NoBG_MMCE))
# sem_contrast_img_promediadas = std_contrast_img_promediadas / np.sqrt(len(contrast_img_promediadas))


# Calcular los intervalos de confianza para cada conjunto de datos
ci_contrast_original = stats.norm.interval(0.95, loc=mean_contrast_original, scale=sem_contrast_original)
ci_contrast_MMCE = stats.norm.interval(0.95, loc=mean_contrast_MMCE, scale=sem_contrast_MMCE)
ci_cir_MMCE = stats.norm.interval(0.95, loc=mean_cir_MMCE, scale=sem_cir_MMCE)
ci_pl_MMCE = stats.norm.interval(0.95, loc=mean_pl_MMCE, scale=sem_pl_MMCE)
ci_contrast_CLAHE = stats.norm.interval(0.95, loc=mean_contrast_CLAHE, scale=sem_contrast_CLAHE)
ci_cir_CLAHE = stats.norm.interval(0.95, loc=mean_cir_CLAHE, scale=sem_cir_CLAHE)
ci_pl_CLAHE = stats.norm.interval(0.95, loc=mean_pl_CLAHE, scale=sem_pl_CLAHE)
ci_contrast_HE = stats.norm.interval(0.95, loc=mean_contrast_HE, scale=sem_contrast_HE)
ci_cir_HE = stats.norm.interval(0.95, loc=mean_cir_HE, scale=sem_cir_HE)
ci_pl_HE = stats.norm.interval(0.95, loc=mean_pl_HE, scale=sem_pl_HE)
ci_contrast_NoBG = stats.norm.interval(0.95, loc=mean_contrast_NoBG, scale=sem_contrast_NoBG)
ci_cir_NoBG = stats.norm.interval(0.95, loc=mean_cir_NoBG, scale=sem_cir_NoBG)
ci_pl_NoBG = stats.norm.interval(0.95, loc=mean_pl_NoBG, scale=sem_pl_NoBG)
ci_contrast_NoBG_CLAHE = stats.norm.interval(0.95, loc=mean_contrast_NoBG_CLAHE, scale=sem_contrast_NoBG_CLAHE)
ci_cir_NoBG_CLAHE = stats.norm.interval(0.95, loc=mean_cir_NoBG_CLAHE, scale=sem_cir_NoBG_CLAHE)
ci_pl_NoBG_CLAHE = stats.norm.interval(0.95, loc=mean_pl_NoBG_CLAHE, scale=sem_pl_NoBG_CLAHE)
ci_contrast_NoBG_MMCE = stats.norm.interval(0.95, loc=mean_contrast_NoBG_MMCE, scale=sem_contrast_NoBG_MMCE)
ci_cir_NoBG_MMCE = stats.norm.interval(0.95, loc=mean_cir_NoBG_MMCE, scale=sem_cir_NoBG_MMCE)
ci_pl_NoBG_MMCE = stats.norm.interval(0.95, loc=mean_pl_NoBG_MMCE, scale=sem_pl_NoBG_MMCE)
# ci_contrast_img_promediadas = stats.norm.interval(0.95, loc=mean_contrast_img_promediadas, scale=sem_contrast_img_promediadas)


# Crear una lista con las medias y los intervalos de confianza
means = [mean_contrast_original, mean_contrast_MMCE, mean_pl_MMCE]
cis = [ci_contrast_original, ci_contrast_MMCE, ci_pl_MMCE]

# Crear el gráfico de barras
fig, ax = plt.subplots()

# Añadir las barras para cada media
bar_labels = ['Contraste Original', 'Contraste MMCE', 'PL MMC']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means, yerr=[abs(top-bot)/2 for top,bot in cis], align='center', alpha=0.5, ecolor='black', capsize=10)

# Añadir el valor de la media a las barras
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height(), str(round(means[i], 2)), ha='center', va='bottom')

# Añadir etiquetas, título y ejes
ax.set_ylabel('Valor Medio')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels)
ax.set_title('Valor medio e intervalo de confianza para cada métrica')
ax.yaxis.grid(True)

# Mostrar el gráfico
plt.tight_layout()
#plt.show()

# Se crea una lista con los valores medios de contraste y los intervalos de confianza
means_contrast = [mean_contrast_original, mean_contrast_MMCE, mean_contrast_CLAHE, mean_contrast_HE, mean_contrast_NoBG, mean_contrast_NoBG_CLAHE, mean_contrast_NoBG_MMCE]
cis_contrast = [ci_contrast_original, ci_contrast_MMCE, ci_contrast_CLAHE, ci_contrast_HE, ci_contrast_NoBG, ci_contrast_NoBG_CLAHE, ci_contrast_NoBG_MMCE]

# Se crea una lista con los valores medios de CIR y los intervalos de confianza
means_cir = [mean_cir_MMCE, mean_cir_CLAHE, mean_cir_HE, mean_cir_NoBG, mean_cir_NoBG_CLAHE, mean_cir_NoBG_MMCE]
cis_cir = [ci_cir_MMCE, ci_cir_CLAHE, ci_cir_HE, ci_cir_NoBG, ci_cir_NoBG_CLAHE, ci_cir_NoBG_MMCE]

# Se crea una lista con los valores medios de PL y los intervalos de confianza
means_pl = [mean_pl_MMCE, mean_pl_CLAHE, mean_pl_HE, mean_pl_NoBG, mean_pl_NoBG_CLAHE, mean_pl_NoBG_MMCE]
cis_pl = [ci_pl_MMCE, ci_pl_CLAHE, ci_pl_HE, ci_pl_NoBG, ci_pl_NoBG_CLAHE, ci_pl_NoBG_MMCE]

# Gráfico de barras para los valores de contraste
fig, ax = plt.subplots()
bar_labels = ['Contraste Original', 'Contraste MMCE', 'Contraste CLAHE', 'Contraste HE', 'Contraste NoBG', 'Contraste NoBG CLAHE', 'Contraste NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_contrast, yerr=[abs(top-bot)/2 for top,bot in cis_contrast], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height(), str(round(means_contrast[i], 2)), ha='center', va='bottom')
ax.set_ylabel('Valor Medio')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels)
ax.set_title('Valor medio e intervalo de confianza para el contraste de cada set de datos')
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de CIR
fig, ax = plt.subplots()
bar_labels = ['CIR MMCE', 'CIR CLAHE', 'CIR HE', 'CIR NoBG', 'CIR NoBG CLAHE', 'CIR NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_cir, yerr=[abs(top-bot)/2 for top,bot in cis_cir], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round(means_cir[i], 2)), ha='center', va='bottom')
ax.set_ylabel('Valor Medio')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels)
ax.set_title('Valor medio e intervalo de confianza para el CIR de cada set de datos')
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de PL
fig, ax = plt.subplots()
bar_labels = ['PL MMCE', 'PL CLAHE', 'PL HE', 'PL NoBG', 'PL NoBG CLAHE', 'PL NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_pl, yerr=[abs(top-bot)/2 for top,bot in cis_pl], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height(), str(round(means_pl[i], 2)), ha='center', va='bottom')
ax.set_ylabel('Valor Medio')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels)
ax.set_title('Valor medio e intervalo de confianza para el PL de cada set de datos')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

# Tabla resumen con DataFrame de pandas
datasets = ['Contraste Original', 'Contraste MMCE', 'PL MMCE']
alphas = [0.05, 0.05, 0.05]
cis2 = [(ci_contrast_original[1]-ci_contrast_original[0])/2, (ci_contrast_MMCE[1]-ci_contrast_MMCE[0])/2, (ci_pl_MMCE[1]-ci_pl_MMCE[0])/2]
# Crear el DataFrame
df_results = pd.DataFrame({
    'Conjunto de datos': datasets,
    'Media': means,
    'Nivel Significativo': alphas,
    'Intervalo de confianza': cis2
})

print(df_results)