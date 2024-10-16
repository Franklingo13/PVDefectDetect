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
archivo_img_promediadas_NoBG_CLAHE = "metricsMeanMaxMin_datasetNoBG_CLAHE.csv"



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
df_img_promediadas = pd.read_csv(os.path.join(path, archivo_img_promediadas))
df_img_promediadas_NoBG_CLAHE = pd.read_csv(os.path.join(path, archivo_img_promediadas_NoBG_CLAHE))

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
# Obtener métricas de las imágenes promediadas
contrast_meanEL = df_img_promediadas["Contraste Mean"].to_list()
cir_meanEL = df_img_promediadas["CIR Mean"].to_list()
pl_meanEL = df_img_promediadas["PL Mean"].to_list()
contrast_maxEL = df_img_promediadas["Contraste Max"].to_list()
cir_maxEL = df_img_promediadas["CIR Max"].to_list()
pl_maxEL = df_img_promediadas["PL Max"].to_list()
contrast_minEL = df_img_promediadas["Contraste Min"].to_list()
cir_minEL = df_img_promediadas["CIR Min"].to_list()
pl_minEL = df_img_promediadas["PL Min"].to_list()
# Obtener métricas de las imágenes promediadas sin fondo con CLAHE
contrast_meanEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["Contrast Mean"].to_list()[0]
cir_meanEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["CIR Mean"].to_list()[0]
pl_meanEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["PL Mean"].to_list()[0]
contrast_maxEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["Contrast Max"].to_list()[0]
cir_maxEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["CIR Max"].to_list()[0]
pl_maxEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["PL Max"].to_list()[0]
contrast_minEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["Contrast Min"].to_list()[0]
cir_minEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["CIR Min"].to_list()[0]
pl_minEL_NoBG_CLAHE = df_img_promediadas_NoBG_CLAHE["PL Min"].to_list()[0]



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
bar_labels = ['Contrast Original', 'Contrast MMCE', 'Contrast CLAHE', 'Contrast HE', 'Contrast NoBG', 'Contrast NoBG CLAHE', 'Contrast NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_contrast, yerr=[abs(top-bot)/2 for top,bot in cis_contrast], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round(means_contrast[i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('Valor Medio', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor medio para el Contrast de cada conjunto de imágenes', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de CIR
fig, ax = plt.subplots()
bar_labels = ['CIR MMCE', 'CIR CLAHE', 'CIR HE', 'CIR NoBG', 'CIR NoBG CLAHE', 'CIR NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_cir, yerr=[abs(top-bot)/2 for top,bot in cis_cir], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round(means_cir[i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('Valor Medio', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor medio para el CIR de cada conjunto de imágenes', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de PL
fig, ax = plt.subplots()
bar_labels = ['PL MMCE', 'PL CLAHE', 'PL HE', 'PL NoBG', 'PL NoBG CLAHE', 'PL NoBG MMCE']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, means_pl, yerr=[abs(top-bot)/2 for top,bot in cis_pl], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round(means_pl[i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('Valor Medio', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor medio para el PL de cada conjunto de imágenes', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# # Gráfico de barras para los valores de contraste de las imágenes promediadas
# fig, ax = plt.subplots()
# bar_labels = ['Contrast Original', 'Contrast Mean', 'Contrast Max', 'Contrast Min']
# x_pos = np.arange(len(bar_labels))
# bars = ax.bar(x_pos, [mean_contrast_original, contrast_meanEL, contrast_maxEL, contrast_minEL], align='center', alpha=0.5, ecolor='black', capsize=10)
# for i in range(len(bars)):
#     ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round([mean_contrast_original, contrast_meanEL, contrast_maxEL, contrast_minEL][i], 2)), ha='center', va='bottom', fontsize=14)
# ax.set_ylabel('Valor Medio', fontsize=14)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(bar_labels, fontsize=12)
# ax.set_title('Valor de Contrast en las imágenes promediadas', fontsize=16)
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

# # Gráfico de barras para los valores de CIR de las imágenes promediadas
# fig, ax = plt.subplots()
# bar_labels = ['CIR Mean', 'CIR Max', 'CIR Min']
# x_pos = np.arange(len(bar_labels))
# bars = ax.bar(x_pos, [cir_meanEL, cir_maxEL, cir_minEL], align='center', alpha=0.5, ecolor='black', capsize=10)
# for i in range(len(bars)):
#     ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round([cir_meanEL, cir_maxEL, cir_minEL][i], 2)), ha='center', va='bottom', fontsize=14)
# ax.set_ylabel('Valor Medio', fontsize=14)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(bar_labels, fontsize=12)
# ax.set_title('Valor de CIR en las imágenes promediadas', fontsize=16)
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

# # Gráfico de barras para los valores de PL de las imágenes promediadas
# fig, ax = plt.subplots()
# bar_labels = ['PL Mean', 'PL Max', 'PL Min']
# x_pos = np.arange(len(bar_labels))
# bars = ax.bar(x_pos, [pl_meanEL, pl_maxEL, pl_minEL], align='center', alpha=0.5, ecolor='black', capsize=10)
# for i in range(len(bars)):
#     ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round([pl_meanEL, pl_maxEL, pl_minEL][i], 2)), ha='center', va='bottom', fontsize=14)
# ax.set_ylabel('Valor Medio', fontsize=14)
# ax.set_xticks(x_pos)
# ax.set_xticklabels(bar_labels, fontsize=12)
# ax.set_title('Valor de PL en las imágenes promediadas', fontsize=16)
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

# Gráfico de barras para los valores de contraste de las imágenes promediadas sin fondo con CLAHE
fig, ax = plt.subplots()
bar_labels = ['C Img. Original', 'C Img. Promedio', 'C Img. Máximos', 'C Img. Mínimos']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, [mean_contrast_original, contrast_meanEL_NoBG_CLAHE, contrast_maxEL_NoBG_CLAHE, contrast_minEL_NoBG_CLAHE], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(
        round([mean_contrast_original, contrast_meanEL_NoBG_CLAHE, contrast_maxEL_NoBG_CLAHE, contrast_minEL_NoBG_CLAHE][i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('Contrast', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor de Contrast en las imágenes promediadas sin fondo con CLAHE', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de CIR de las imágenes promediadas sin fondo con CLAHE
fig, ax = plt.subplots()
bar_labels = ['CIR Img. Promedio', 'CIR Img. Máximos', 'CIR Img. Mínimos']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, [cir_meanEL_NoBG_CLAHE, cir_maxEL_NoBG_CLAHE, cir_minEL_NoBG_CLAHE], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round([cir_meanEL_NoBG_CLAHE, cir_maxEL_NoBG_CLAHE, cir_minEL_NoBG_CLAHE][i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('CIR', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor de CIR en las imágenes promediadas sin fondo con CLAHE', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
#plt.show()

# Gráfico de barras para los valores de PL de las imágenes promediadas sin fondo con CLAHE
fig, ax = plt.subplots()
bar_labels = ['PL Img. Promedio', 'PL Img. Máximos', 'PL Img. Mínimos']
x_pos = np.arange(len(bar_labels))
bars = ax.bar(x_pos, [pl_meanEL_NoBG_CLAHE, pl_maxEL_NoBG_CLAHE, pl_minEL_NoBG_CLAHE], align='center', alpha=0.5, ecolor='black', capsize=10)
for i in range(len(bars)):
    ax.text(bars[i].get_x() + bars[i].get_width(), bars[i].get_height(), str(round([pl_meanEL_NoBG_CLAHE, pl_maxEL_NoBG_CLAHE, pl_minEL_NoBG_CLAHE][i], 2)), ha='center', va='bottom', fontsize=20)
ax.set_ylabel('PL', fontsize=22)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=20)
ax.set_title('Valor de PL en las imágenes promediadas sin fondo con CLAHE', fontsize=20)
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()



# Tabla resumen con DataFrame de pandas sobre la métrica de contraste
datasets_contraste = ['Contraste Original', 'Contraste MMCE', 'Contraste CLAHE', 'Contraste HE', 'Contraste NoBG', 'Contraste NoBG CLAHE', 'Contraste NoBG MMCE']
alphas_contraste = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
cis_contraste = [(ci_contrast_original[1]-ci_contrast_original[0])/2, (
    ci_contrast_MMCE[1]-ci_contrast_MMCE[0])/2, (ci_contrast_CLAHE[1]-ci_contrast_CLAHE[0])/2, (ci_contrast_HE[1]-ci_contrast_HE[0])/2, (ci_contrast_NoBG[1]-ci_contrast_NoBG[0])/2, (ci_contrast_NoBG_CLAHE[1]-ci_contrast_NoBG_CLAHE[0])/2, (ci_contrast_NoBG_MMCE[1]-ci_contrast_NoBG_MMCE[0])/2]
# Crear el DataFrame
df_results_contraste = pd.DataFrame({
    'Conjunto de datos': datasets_contraste,
    'Media': means_contrast,
    'Nivel Significativo': alphas_contraste,
    'Intervalo de confianza': cis_contraste
})
print(df_results_contraste)

# Tabla resumen con DataFrame de pandas sobre la métrica de CIR
datasets_cir = ['CIR MMCE', 'CIR CLAHE', 'CIR HE', 'CIR NoBG', 'CIR NoBG CLAHE', 'CIR NoBG MMCE']
alphas_cir = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
cis_cir = [(ci_cir_MMCE[1]-ci_cir_MMCE[0])/2, (ci_cir_CLAHE[1]-ci_cir_CLAHE[0])/2, (ci_cir_HE[1]-ci_cir_HE[0])/2, (ci_cir_NoBG[1]-ci_cir_NoBG[0])/2, (ci_cir_NoBG_CLAHE[1]-ci_cir_NoBG_CLAHE[0])/2, (ci_cir_NoBG_MMCE[1]-ci_cir_NoBG_MMCE[0])/2]
# Crear el DataFrame
df_results_cir = pd.DataFrame({
    'Conjunto de datos': datasets_cir,
    'Media': means_cir,
    'Nivel Significativo': alphas_cir,
    'Intervalo de confianza': cis_cir
})
print(df_results_cir)

# Tabla resumen con DataFrame de pandas sobre la métrica de PL
datasets_pl = ['PL MMCE', 'PL CLAHE', 'PL HE', 'PL NoBG', 'PL NoBG CLAHE', 'PL NoBG MMCE']
alphas_pl = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
cis_pl = [(ci_pl_MMCE[1]-ci_pl_MMCE[0])/2, (ci_pl_CLAHE[1]-ci_pl_CLAHE[0])/2, (ci_pl_HE[1]-ci_pl_HE[0])/2, (ci_pl_NoBG[1]-ci_pl_NoBG[0])/2, (ci_pl_NoBG_CLAHE[1]-ci_pl_NoBG_CLAHE[0])/2, (ci_pl_NoBG_MMCE[1]-ci_pl_NoBG_MMCE[0])/2]
# Crear el DataFrame
df_results_pl = pd.DataFrame({
    'Conjunto de datos': datasets_pl,
    'Media': means_pl,
    'Nivel Significativo': alphas_pl,
    'Intervalo de confianza': cis_pl
})
print(df_results_pl)