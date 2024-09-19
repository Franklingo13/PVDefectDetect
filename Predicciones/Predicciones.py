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
images = [cv.imread(file) for file in list_images(imgs_path)]
print(f'Número de imágenes: {len(images)}')

# Crear las carpetas de salida si no existen
os.makedirs(f'{out_path}/ann', exist_ok=True)
os.makedirs(f'{out_path}/image', exist_ok=True)

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
modelhandler.load_model(weight_path)

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
    cv.imwrite(f'{out_path}/ann/annImage{idx}.png', annImage)
    cv.imwrite(f'{out_path}/image/Image{idx}.png', images[idx])

# Nombres de las clases
class_names = ['busbar', 'crack', 'dark']
# Número de clases en el conjunto de datos
n_classes = 3  

# Generar la imagen del panel completo con las predicciones
panel_prediction_img = combine_panel_predictions(images, masks_each, cols =6)

# Mostrar o guardar la imagen del panel
#panel_prediction_img.show()  
panel_prediction_img.save(f'{out_path}/panel_predictions_mono.png')

# Inicializar una lista para almacenar las estadísticas
stats = []

# Iterar sobre todas las imágenes y usar CrackCell para extraer las estadísticas
for idx in range(len(masks_each)):
    img_path = f'{out_path}/image/Image{idx}.png'
    ann_path = f'{out_path}/ann/annImage{idx}.png'

    # Generar estadísticas de CrackCell
    stats.append(
        generate_crackcell_stats(img_path, ann_path, busbar_num=n_busbar, crack_inx=100, busbar_inx=10))
    

# Convertir la lista de estadísticas en un DataFrame de pandas
crackcell_stats = pd.DataFrame(stats)
# Mostrar el DataFrame
#print(crackcell_stats)
area_statistics = generate_area_percentage_statistics(masks_each, class_names)

generate_defect_bar_chart(area_statistics, f'{out_path}/defect_bar_chart.png')
generate_average_area_bar_chart(area_statistics, f'{out_path}/average_area_bar_chart.png')
generate_crackcell_stats_bar_chart(crackcell_stats, f'{out_path}/crackcell_stats_bar_chart.png')
# Guardar las estadísticas en un archivo CSV
save_statistics(area_statistics, f'{out_path}/area_statistics.csv')
save_statistics(crackcell_stats, f'{out_path}/crackcell_statistics.csv')

# Generar y guardar los mapas de calor de las máscaras predichas
save_heatmap_images(masks_each, out_path)

# Generar la matriz de coocurrencia
cooccurrence_matrix = generate_cooccurrence_matrix(masks_each, n_classes)
plot_cooccurrence_matrix(cooccurrence_matrix, class_names, save_path=f'{out_path}/cooccurrence_matrix.png')

## Introducciones ##
# Almacena el texto explicativo en variables separadas
intro_mapas_calor = (
    "Un mapa de calor de distribución de predicciones proporciona "
    "una visualización de cómo se presentan los defectos en "
    "las celdas del panel, revelando las áreas con mayor "
    "frecuencia de fallos. Este gráfico facilita la identificación "
    "de zonas críticas y patrones de deterioro en el panel fotovoltaico. "
    "Además, permite un monitoreo efectivo del estado del panel a lo "
    "largo del tiempo, apoya la toma de decisiones en el mantenimiento "
    "y ofrece una herramienta útil para la comparación y evaluación "
    "eficiente entre diferentes paneles."
)

intro_cooccurrence_matrix = (
    "Una Matriz de Coocurrencia muestra la frecuencia con la " 
    "que diferentes clases aparecen juntas en las imágenes " 
    "analizadas. Este análisis ayuda a identificar " 
    "posibles interacciones entre las clases, facilitando " 
    "una mejor comprensión del comportamiento de las " 
    "predicciones del modelo y de los defectos en el panel. "
)

intro_crackcell_stats = (
    "La tabla presenta un resumen de las características extraídas de las celdas fotovoltaicas "
    "con defectos. Se incluyen los siguientes parámetros: <br/><br/>"
    "Área Inactiva (%): Muestra el porcentaje de la celda que se considera incapaz de contribuir "
    "efectivamente a la generación de energía. Estas zonas inactivas se definen como aquellas áreas "
    "que están cubiertas por grietas que interrumpen el flujo de corriente, especialmente en las "
    "cercanías de las barras colectoras. A diferencia de las zonas oscuras (Dark) predichas por el "
    "modelo, este indicador se enfoca en las áreas separadas del panel debido a la combinación de grietas "
    "y barras colectoras, permitiendo evaluar la extensión del daño en la celda. <br/><br/>"
    "Longitud de Grieta (píxeles): Mide la longitud total de las grietas presentes en la celda. Esta métrica "
    "permite identificar la severidad del daño estructural en la celda. <br/><br/>"
    "Brillo: Calcula el nivel promedio de brillo en las áreas inactivas de la celda, proporcionando una medida "
    "adicional del estado de deterioro de la celda. Un valor de brillo igual a 1 indica que la celda no presenta "
    "grietas, mientras que valores más bajos podrían sugerir un mayor deterioro. "
)

explain_crackcell_stats = (
    "Para esta tabla primero se obtiene el trazado de las grietas y las barras colectoras, lo que facilita "
    "la medición precisa de la longitud de las grietas. En lugar de considerar todos los píxeles predichos "
    "por el modelo, solo se toman en cuenta aquellos que realmente influyen en la longitud de la grieta. "
    "Además, al combinar esta información con las posiciones de las barras colectoras, se puede estimar las "
    "zonas de la celda que probablemente enfrentarán dificultades para generar energía. "
)

intro_area_statistics = (
    "La tabla de estadísticas generada a partir de las predicciones muestra el porcentaje "
    "de área que ocupa cada clase dentro de cada celda fotovoltaica, en relación con el área "
    "total de la imagen. Los elementos incluidos son: <br/><br/>"
    "Dark [%]: Indica el porcentaje de la imagen cubierto por zonas oscuras. Estas zonas, "
    "visibles en las imágenes de electroluminiscencia, señalan áreas del panel que "
    "están desconectadas y, por lo tanto, no contribuyen a la generación de energía. <br/><br/>"
    "Busbar [%]: Indica el porcentaje de la imagen ocupado por las barras colectoras, " 
    "componentes que transportan la corriente eléctrica generada por la celda. <br/><br/>"
    "Crack [%]: Mide el porcentaje de la imagen afectado por grietas. Las grietas en " 
    "las celdas pueden interrumpir la continuidad eléctrica, afectando la eficiencia del panel."
)

explain_area_statistics = (
    "Para interpretar correctamente esta tabla, se deben analizar los porcentajes de área ocupados "
    "por cada clase en cada imagen. Un porcentaje elevado en la clase `Dark` podría indicar problemas "
    "serios en la funcionalidad de la celda, ya que sugiere que una parte significativa del panel está "
    "inactiva. Un aumento en el porcentaje de `Crack` puede ser un indicio de un deterioro estructural "
    "progresivo de la celda, lo cual podría comprometer su eficiencia. La clase `Busbar` debe mantenerse "
    "estable, y cualquier variación significativa podría indicar problemas en la predicción del modelo o "
    "en la integridad física de las barras colectoras. "
)
panel_predictions_text = (
    "La imagen muestra el panel completo con las imágenes originales y las predicciones superpuestas."
)

intro_text = (
    "Este reporte presenta los resultados del análisis de imágenes de electroluminiscencia (EL) "
    "de un panel fotovoltaico. Para ello se cuenta con un modelo de red neuronal convolucional (CNN) "
    "entrenado para detectar defectos en las celdas solares, como grietas y zonas oscuras, así como "
    "barras colectoras. "
    " Estas características del panel se asignan a 3 clases: `Busbar`, `Crack` y `Dark`. "
    "Se incluyen visualizaciones de las predicciones del modelo, "
    "estadísticas de área por imagen, una matriz de coocurrencia y mapas de calor de las predicciones. "
    "Además, se presentan estimaciones a partir de las predicciones, que incluyen el porcentaje de área inactiva, "
    "la longitud de grietas y el nivel de brillo en las celdas con defectos. "
    )

average_area_bar_chart_text = (
    "El gráfico de barras muestra el área promedio ocupada por cada clase en las celdas fotovoltaicas. "
)

defect_bar_chart_text = (
    "El gráfico de barras muestra el número de celdas que presentan uno o más defectos. "
    "Se muestran cuatro categorías: celdas con grietas y zonas oscuras, celdas solo con zonas oscuras, "
    "celdas solo con grietas y celdas intactas. "
)

## Crear el reporte ##
## Generación del reporte en PDF
fecha_actual = datetime.now().strftime("%Y%m%d")
reporte_pdf = f'{out_path}/reporte_analisis_EL_{fecha_actual}_{ID_panel}.pdf'

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

    # # Añadir el índice de contenido
    # toc = tableofcontents.TableOfContents()
    # toc.levelStyles = [
    #     ParagraphStyle(fontName='Helvetica-Bold', fontSize=14, name='Heading1'),
    #     ParagraphStyle(fontSize=12, name='Heading2')
    # ]
    # elements.append(Paragraph("Índice", styles['Title']))
    # elements.append(toc)
    # elements.append(Spacer(1, 12))

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