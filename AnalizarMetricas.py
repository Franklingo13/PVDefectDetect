import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from ImagePreprocessing.utils import *

## Script para analizar las métricas obtenidas en el archivo CSV

path = read_folder_path(r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\PVDefectDetect\ImagePreprocessing\Salidas\TestContraste")
nombre_archivo = "metricsMMC.csv"


contrast_original = []  # Lista para almacenar los valores de contraste del dataset original
contrast_MMCE = []  # Lista para almacenar los valores de contraste del dataset con MMC
cir_MMCE =[]  # Lista para almacenar los valores de CIR del dataset con MMC
pl_MMCE = []  # Lista para almacenar los valores de PL del dataset con MMC

# Leer el archivo CSV
df = pd.read_csv(os.path.join(path, nombre_archivo))

# Obtener los valores de contraste, PL y CIR
contrast_original = df["Contraste EL"].to_list()
contrast_MMCE = df["Contraste MMC"].to_list()
cir_MMCE = df["CIR MMC"].to_list()
pl_MMCE = df["PL MMC"].to_list()

# Cálculo de las medias de los valores de contraste, PL y CIR
mean_contrast_original = np.mean(contrast_original)
mean_contrast_MMCE = np.mean(contrast_MMCE)
mean_cir_MMCE = np.mean(cir_MMCE)
mean_pl_MMCE = np.mean(pl_MMCE)

# Cálculo de la desviación estándar de los valores de contraste, PL y CIR
std_contrast_original = np.std(contrast_original)
std_contrast_MMCE = np.std(contrast_MMCE)
std_cir_MMCE = np.std(cir_MMCE)
std_pl_MMCE = np.std(pl_MMCE)

# Cálculo del error estándar de los valores de contraste, PL y CIR
sem_contrast_original = std_contrast_original / np.sqrt(len(contrast_original))
sem_contrast_MMCE = std_contrast_MMCE / np.sqrt(len(contrast_MMCE))
sem_cir_MMCE = std_cir_MMCE / np.sqrt(len(cir_MMCE))
sem_pl_MMCE = std_pl_MMCE / np.sqrt(len(pl_MMCE))


# Calcular los intervalos de confianza para cada conjunto de datos
ci_contrast_original = stats.norm.interval(0.95, loc=mean_contrast_original, scale=sem_contrast_original)
ci_contrast_MMCE = stats.norm.interval(0.95, loc=mean_contrast_MMCE, scale=sem_contrast_MMCE)
ci_cir_MMCE = stats.norm.interval(0.95, loc=mean_cir_MMCE, scale=sem_cir_MMCE)
ci_pl_MMCE = stats.norm.interval(0.95, loc=mean_pl_MMCE, scale=sem_pl_MMCE)


# Crear una lista con las medias y los intervalos de confianza
means = [mean_contrast_original, mean_contrast_MMCE, mean_pl_MMCE]
cis = [ci_contrast_original, ci_contrast_MMCE, ci_pl_MMCE]

# Crear el gráfico de barras
fig, ax = plt.subplots()

# Añadir las barras para cada media
bar_labels = ['Contraste Original', 'Contraste MMC', 'PL MMC']
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