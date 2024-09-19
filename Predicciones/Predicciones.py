# Importar librerías

import torch 
from torch.nn import DataParallel
from torchvision.utils import draw_segmentation_masks
import cv2 as cv
import numpy as np
from unet_model import construct_unet
from imutils.paths import list_images
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import json
# Importar Model Handler
from pv_vision.nn import ModelHandler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline    # Para Jupyter Notebookutilizar matplotlib.pyplot.show() en lugar

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

from Funciones import *

## Parámetros ##

# # Carpeta de imágenes
# imgs_path = 'D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/Datasets_EL/CeldasIndividuales/Mono2_V40_I5_t28'
# images = [cv.imread(file) for file in list_images(imgs_path)]           #imágenes de celdas
# # Verificar que las imágenes se cargaron correctamente
# print(f'Número de imágenes: {len(images)}')
# weight_path = 'D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/Predicciones/PesosGColab/unetv32.pt'
# out_path='D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/Predicciones/SalidasMonoV4'
# os.makedirs(f'{out_path}/ann', exist_ok=True)
# os.makedirs(f'{out_path}/image', exist_ok=True)
# n_busbar = 2
# ID_panel = 'Mono2_V40_I5_t28'

def select_folder():
    folder_selected = filedialog.askdirectory()
    return folder_selected

def select_file():
    file_selected = filedialog.askopenfilename()
    return file_selected

def submit():
    global imgs_path, weight_path, out_path, n_busbar, ID_panel
    imgs_path = entry_imgs_path.get()
    weight_path = entry_weight_path.get()
    out_path = entry_out_path.get()
    n_busbar = entry_n_busbar.get()
    ID_panel = entry_ID_panel.get()
    
    # Validar las entradas
    if not imgs_path or not weight_path or not out_path or not n_busbar or not ID_panel:
        messagebox.showerror("Error", "Todos los campos son obligatorios")
        return
    
    try:
        n_busbar = int(n_busbar)
    except ValueError:
        messagebox.showerror("Error", "El número de barras colectoras debe ser un número entero")
        return
    
    root.destroy()

root = tk.Tk()
root.title("Parámetros del Análisis de Imágenes EL")

# Carpeta de imágenes
tk.Label(root, text="Carpeta de Imágenes:").grid(row=0, column=0, padx=10, pady=5)
entry_imgs_path = tk.Entry(root, width=50)
entry_imgs_path.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_imgs_path.insert(0, select_folder())).grid(row=0, column=2, padx=10, pady=5)

# Archivo de pesos del modelo
tk.Label(root, text="Archivo de Pesos del Modelo:").grid(row=1, column=0, padx=10, pady=5)
entry_weight_path = tk.Entry(root, width=50)
entry_weight_path.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_weight_path.insert(0, select_file())).grid(row=1, column=2, padx=10, pady=5)

# Carpeta de salida
tk.Label(root, text="Carpeta de Salida:").grid(row=2, column=0, padx=10, pady=5)
entry_out_path = tk.Entry(root, width=50)
entry_out_path.grid(row=2, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_out_path.insert(0, select_folder())).grid(row=2, column=2, padx=10, pady=5)

# Número de barras colectoras
tk.Label(root, text="Número de Barras Colectoras:").grid(row=3, column=0, padx=10, pady=5)
entry_n_busbar = tk.Entry(root, width=50)
entry_n_busbar.grid(row=3, column=1, padx=10, pady=5)

# ID del panel
tk.Label(root, text="ID del Panel:").grid(row=4, column=0, padx=10, pady=5)
entry_ID_panel = tk.Entry(root, width=50)
entry_ID_panel.grid(row=4, column=1, padx=10, pady=5)

# Botón de enviar
tk.Button(root, text="Enviar", command=submit).grid(row=5, column=1, padx=10, pady=20)

root.mainloop()

# Verificar que las imágenes se cargaron correctamente
images = [cv.imread(file) for file in list_images(imgs_path)] # type: ignore
print(f'Número de imágenes: {len(images)}')

# Crear las carpetas de salida si no existen
os.makedirs(f'{out_path}/ann', exist_ok=True)  # type: ignore
os.makedirs(f'{out_path}/image', exist_ok=True)  # type: ignore

## Cargar modelo ##

# Crea el dataset
imgset = myDataset(images, transform)

# Si hay una GPU disponible, se utilizará.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Dispositivo: {device}')
# Crea una instancia del modelo U-Net con 5 canales de salida. 
# Número de canales de salida = al número de clases
unet = construct_unet(5)
# Se "envuelve" el modelo en un objeto DataParallel. 
# Esto permite que el modelo se ejecute en paralelo en múltiples GPUs, si están disponibles.
unet = DataParallel(unet)

# Inicia el manejador del modelo (ModelHandler). 
# Este objeto se encargará de la gestión del modelo, incluyendo la carga de los datos, 
# la ejecución del modelo y el almacenamiento de los resultados.
modelhandler = ModelHandler(
    # El modelo que se va a utilizar.
    model=unet,
    #model_output='out_deeplab',
    # El conjunto de datos que se utilizará para las pruebas.
    test_dataset=imgset,
    # Indica que sólo se realizarán predicciones, no se entrenará el modelo.
    predict_only=True,
    # El tamaño del lote que se utilizará durante la validación. 
    batch_size_val=2,
    # El dispositivo en el que se ejecutará el modelo.
    device=device,
    # El directorio donde se guardarán los resultados. 
    save_dir = 'D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/Predicciones/SalidasMono/logs',
    # El nombre que se utilizará para guardar los resultados. 
    save_name='Unetv32_cell_prediction_val'
)
# Cargar los pesos del modelo desde el archivo especificado por 'weight_path'.
modelhandler.load_model(weight_path)  # type: ignore

# Ejecución del modelo en el conjunto de datos.
# Esto generará predicciones para cada imagen en el conjunto de datos.
masks = modelhandler.predict(save=True)
masks_each = get_masks(masks)
print('Predicciones realizadas')

# Creación de imagen de anotaciones `annImage` con las máscaras `crack` y 
# `busbar` de las predicciones
class_values = [0, 10, 100]
for idx, mask in enumerate(masks_each):
    annImage = np.zeros(mask[0].shape, dtype=np.uint8)
    annImage[mask[0]] = class_values[1]  # busbar
    annImage[mask[1]] = class_values[2]  # crack

    annImage = cv.resize(annImage, (images[idx].shape[1], images[idx].shape[0]), 
                         interpolation=cv.INTER_NEAREST)
    cv.imwrite(f'{out_path}/ann/annImage{idx}.png', annImage)  # type: ignore
    cv.imwrite(f'{out_path}/image/Image{idx}.png', images[idx])  # type: ignore

# Nombres de las clases
class_names = ['busbar', 'crack', 'dark']
# Número de clases en el conjunto de datos
n_classes = 3  

# Generar la imagen del panel completo con las predicciones
panel_prediction_img = combine_panel_predictions(images, masks_each, cols =6)

# Mostrar o guardar la imagen del panel
#panel_prediction_img.show()  
panel_prediction_img.save(f'{out_path}/panel_predictions_mono.png')  # type: ignore

# Inicializar una lista para almacenar las estadísticas
stats = []

# Iterar sobre todas las imágenes y usar CrackCell para extraer las estadísticas
for idx in range(len(masks_each)):
    img_path = f'{out_path}/image/Image{idx}.png'  # type: ignore
    ann_path = f'{out_path}/ann/annImage{idx}.png'  # type: ignore

    # Generar estadísticas de CrackCell
    stats.append(
        generate_crackcell_stats(
            img_path, ann_path, busbar_num=n_busbar, crack_inx=100, busbar_inx=10))  # type: ignore
    

# Convertir la lista de estadísticas en un DataFrame de pandas
crackcell_stats = pd.DataFrame(stats)
# Mostrar el DataFrame
#print(crackcell_stats)
area_statistics = generate_area_percentage_statistics(masks_each, class_names)

generate_defect_bar_chart(area_statistics, f'{out_path}/defect_bar_chart.png')  # type: ignore
generate_average_area_bar_chart(area_statistics, f'{out_path}/average_area_bar_chart.png')  # type: ignore
generate_crackcell_stats_bar_chart(crackcell_stats, f'{out_path}/crackcell_stats_bar_chart.png')    # type: ignore
# Guardar las estadísticas en un archivo CSV
save_statistics(area_statistics, f'{out_path}/area_statistics.csv') # type: ignore
save_statistics(crackcell_stats, f'{out_path}/crackcell_statistics.csv')    # type: ignore

# Generar y guardar los mapas de calor de las máscaras predichas
save_heatmap_images(masks_each, out_path)   # type: ignore

# Generar la matriz de coocurrencia
cooccurrence_matrix = generate_cooccurrence_matrix(masks_each, n_classes)
plot_cooccurrence_matrix(
    cooccurrence_matrix, class_names, save_path=f'{out_path}/cooccurrence_matrix.png')  # type: ignore

## Introducciones ##
# Leer el archivo de texto y cargar el contenido en un diccionario
# Obtener la ruta del directorio actual
current_dir = os.path.dirname(__file__)

# Construir la ruta completa al archivo de texto
file_path = os.path.join(current_dir, 'textos_reporte.txt')

# Leer el archivo de texto y cargar el contenido en un diccionario
with open(file_path, 'r', encoding='utf-8') as file:
    textos_reporte = json.load(file)
# Acceder a los textos desde el diccionario
intro_mapas_calor = textos_reporte["intro_mapas_calor"]
intro_cooccurrence_matrix = textos_reporte["intro_cooccurrence_matrix"]
intro_crackcell_stats = textos_reporte["intro_crackcell_stats"]
explain_crackcell_stats = textos_reporte["explain_crackcell_stats"]
intro_area_statistics = textos_reporte["intro_area_statistics"]
explain_area_statistics = textos_reporte["explain_area_statistics"]
panel_predictions_text = textos_reporte["panel_predictions_text"]
intro_text = textos_reporte["intro_text"]
average_area_bar_chart_text = textos_reporte["average_area_bar_chart_text"]
defect_bar_chart_text = textos_reporte["defect_bar_chart_text"]

## Crear el reporte ##
## Generación del reporte en PDF
fecha_actual = datetime.now().strftime("%Y%m%d")
reporte_pdf = f'{out_path}/reporte_analisis_EL_{fecha_actual}_{ID_panel}.pdf'   # type: ignore

# Función para generar el reporte en PDF
def generate_pdf_report():
    pdf = SimpleDocTemplate(reporte_pdf, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Crear un estilo de párrafo justificado
    justified_style = ParagraphStyle(
        name='Justified',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY
    )

    # Añadir título
    title = Paragraph("Reporte de Resultados de Análisis de Imágenes EL", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Añadir introducción, explicando el contenido del reporte
    elements.append(Paragraph(intro_text, justified_style))
    elements.append(Spacer(1, 12))

    # Añadir la imagen del panel completo con las predicciones
    elements.append(Paragraph("Panel de Predicciones", styles['Heading2']))
    elements.append(Paragraph(panel_predictions_text, justified_style))
    elements.append(RLImage(f'{out_path}/panel_predictions_mono.png', width=500, height=630))
    print(f"Panel de Predicciones agregado al reporte")

    # Añadir gráfico de barras de defectos
    elements.append(Paragraph("Distribución de Defectos", styles['Heading2']))
    #elements.append(Paragraph("Número de Celdas por Clase Predicha", styles['Heading3']))
    elements.append(Paragraph(defect_bar_chart_text, justified_style))
    elements.append(RLImage(f'{out_path}/defect_bar_chart.png', width=400, height=300))
    print(f"Gráfico de barras de defectos agregado al reporte")

    # Añadir gráfico de barras de área promedio
    elements.append(Paragraph("Área Promedio Ocupada por Cada Clase Predicha", styles['Heading2']))
    #elements.append(Paragraph("Área Promedio Ocupada por Cada Clase Predicha", styles['Heading3']))
    elements.append(Paragraph(average_area_bar_chart_text, justified_style))
    elements.append(RLImage(f'{out_path}/average_area_bar_chart.png', width=400, height=300))
    print(f"Gráfico de barras de área promedio agregado al reporte")

    # Añadir estadísticas de porcentaje de área por imagen
    elements.append(Paragraph("Estadísticas de Porcentaje de Área por Imagen", styles['Heading2']))
    elements.append(Paragraph(intro_area_statistics, justified_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(explain_area_statistics, justified_style))
    elements.append(Spacer(1, 12))
    elements.append(create_table(area_statistics))
    print(f"Estadísticas de Porcentaje de Área por Imagen agregadas al reporte")

    # Agregar la matriz de coocurrencia al PDF
    elements.append(Paragraph("Matriz de Coocurrencia", styles['Heading2']))
    elements.append(Paragraph(intro_cooccurrence_matrix, justified_style))
    elements.append(RLImage(f'{out_path}/cooccurrence_matrix.png', width=400, height=300))
    elements.append(Spacer(1, 12))
    print(f"Matriz de coocurrencia agregada al reporte")

    # Agregar los mapas de calor al PDF
    elements.append(Paragraph("Distribución Espacial de las Predicciones en Mapas de Calor", styles['Heading2']))
    elements.append(Paragraph(intro_mapas_calor, justified_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Mapa de Calor de Barras Colectoras", styles['Heading3']))
    elements.append(RLImage(f'{out_path}/heatmap_busbar.png', width=200, height=200))
    elements.append(Paragraph("Mapa de Calor de Grietas", styles['Heading3']))
    elements.append(RLImage(f'{out_path}/heatmap_crack.png', width=200, height=200))
    elements.append(Paragraph("Mapa de Calor de Zonas Oscuras", styles['Heading3']))
    elements.append(RLImage(f'{out_path}/heatmap_dark.png', width=200, height=200))
    print(f"Mapas de calor agregados al reporte")

    # Añadir estadísticas de CrackCell
    elements.append(Paragraph("Estimaciones de Área Inactiva, Longitud de Grietas y Nivel de Brillo", styles['Heading2']))
    elements.append(Paragraph(intro_crackcell_stats, justified_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(explain_crackcell_stats, justified_style))
    elements.append(Spacer(1, 12))
    elements.append(create_table(crackcell_stats))
    print(f"Estadísticas de CrackCell agregadas al reporte")

    # Añadir gráfico de barras de estadísticas de CrackCell
    elements.append(Paragraph("Gráfico de Barras de las Estimaciones", styles['Heading2']))
    elements.append(Paragraph("Valores Promedio de las Estimaciones", styles['Heading3']))
    elements.append(RLImage(f'{out_path}/crackcell_stats_bar_chart.png', width=400, height=300))
    print(f"Gráfico de barras de estadísticas de CrackCell agregado al reporte")

    pdf.build(elements)

# Generar el reporte en PDF
generate_pdf_report()
print(f"Reporte PDF generado en: {reporte_pdf}")