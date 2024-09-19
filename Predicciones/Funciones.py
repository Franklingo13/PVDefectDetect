# Importar librerías

import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd

from pv_vision.crack_analysis.crackcell import CrackCell
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
### FUNCIONES

# Definir la transformación de las imágenes que se pasará al manejador del modelo
transform = transforms.Compose([
    # Convertir la imagen a un tensor de PyTorch y escalar los valores de los píxeles entre 0 y 1
    transforms.ToTensor(),
    # Normalizar cada canal de color de la imagen. 
    # Los valores de la media y la desviación estándar se especifican para cada canal (RGB). 
    # Estos valores son los valores de media y desviación estándar del conjunto de datos ImageNet.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Definir una clase personalizada que hereda de Dataset
class myDataset(Dataset):
    """ Clase para cargar un conjunto de datos de imágenes
    Args:
        images (list): Lista de imágenes
        transform (callable): Transformación a aplicar a las imágenes
    """
    # El método de inicialización se llama cuando se crea una instancia de la clase
    def __init__(self, images, transform):
        # Guardar las imágenes y la transformación como atributos de la instancia
        self.images = images
        self.transform = transform

    # El método __len__ devuelve el número de elementos en el conjunto de datos
    def __len__(self):
        return len(self.images)

    # El método __getitem__ se utiliza para obtener un elemento del conjunto de datos
    def __getitem__(self, idx):
        # Redimensionar la imagen al tamaño deseado
        image = cv.resize(self.images[idx], (256, 256))
        # Aplicar la transformación a la imagen
        image = self.transform(image)
        
        # Devolver la imagen transformada
        return image

# Función para obtener las máscaras de las predicciones del modelo.
def get_masks(masks_raw):
    """ Obtiene las máscaras de segmentación a partir de las predicciones del modelo.

    Args:
        masks_raw (np.ndarray): Predicciones del modelo.

    Returns:
        mask_each (list): Lista de máscaras de segmentación.
    """
    # Se creó una lista vacía para almacenar las máscaras.
    masks_each = []
    # Se aplicó la función softmax a las predicciones del modelo y se obtuvo la clase 
    # con la mayor probabilidad para cada píxel.
    masks_all = torch.nn.functional.softmax(torch.from_numpy(masks_raw), dim=1).argmax(dim=1)
    # Para cada máscara en masks_all, se crearon máscaras booleanas para cada clase 
    # y se añadieron a la lista masks_each.
    for masks in masks_all:
        busbar = masks==1
        crack = masks==2
        dark = masks==4
        masks_each.append(torch.dstack([busbar, crack, dark]).permute(2, 0, 1))
    return masks_each

# Se definió una función para dibujar las máscaras sobre las imágenes.
def draw_mask(img, masks, alpha=0.6):
    """ Dibuja las máscaras de segmentación sobre la imagen de entrada.

    Args:
        img (PIL.Image): Imagen de entrada.
        masks (list): Lista de máscaras de segmentación.
        alpha (float): Opacidad de las máscaras.

    Returns:
        PIL.Image: Imagen con las máscaras dibujadas.
    """

    # Mapa de colores para las diferentes clases de máscaras.
    colors = {
        'dark': (68, 114, 148),
        'crack': (165, 59, 63),
        'busbar': (222, 156, 83)
    }
    # Se convirtió la imagen a un objeto de la clase Image de PIL y se redimensionó.
    img = Image.fromarray(img)
    # Transformación para redimensionar las imágenes a 256x256 y convertirlas a tensores de PyTorch.
    resize = transforms.Compose([transforms.Resize((256, 256)), transforms.PILToTensor()])
    img = resize(img)
    # Se dibujaron las máscaras sobre la imagen con la opacidad especificada y se devolvió la imagen resultante.
    combo = draw_segmentation_masks(
        img, masks, alpha=alpha, colors=[colors[key] for key in ['busbar', 'crack', 'dark']]) # type: ignore
    return F.to_pil_image(combo)

# Mapas de calor de las máscaras predichas
def generate_heatmap(masks_each, class_index):
    """
    Genera un mapa de calor para una clase específica combinando todas las predicciones.

    Args:
        masks_each (list): Lista de máscaras predichas para cada imagen.
        class_index (int): Índice de la clase para la cual se desea generar el mapa de calor.
                           Por ejemplo, 0 para 'busbar', 1 para 'crack', 3 para 'dark'.

    Returns:
        np.ndarray: Mapa de calor normalizado.
    """
    # Sumar todas las máscaras correspondientes a la clase
    heatmap = np.sum([mask[class_index] for mask in masks_each], axis=0)
    
    # Normalizar el mapa de calor para que los valores estén entre 0 y 1
    #heatmap = combined_mask / combined_mask.max()
    
    return heatmap


# Función para generar y mostrar una matriz de coocurrencia
def generate_cooccurrence_matrix(masks_each, n_classes):
    """
    Genera una matriz de coocurrencia para las clases en las imágenes.

    Args:
        masks_each (list): Lista de máscaras predichas para cada imagen.
        n_classes (int): Número de clases en el conjunto de datos.

    Returns:
        np.ndarray: Matriz de coocurrencia.
    """
    cooccurrence_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for masks in masks_each:
        # Convertir a NumPy si es un tensor de PyTorch
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # Crear un conjunto de clases presentes en la imagen
        present_classes = set()
        for class_index in range(n_classes):
            if np.any(masks[class_index]):
                present_classes.add(class_index)
        
        # Actualizar la matriz de coocurrencia
        for class1 in present_classes:
            for class2 in present_classes:
                cooccurrence_matrix[class1, class2] += 1

    return cooccurrence_matrix

def save_cooccurrence_matrix(cooccurrence_matrix, class_names, file_path):
    """
    Guarda la matriz de coocurrencia en un archivo CSV.

    Args:
        cooccurrence_matrix (np.ndarray): Matriz de coocurrencia.
        class_names (list): Lista de nombres de las clases.
        file_path (str): Ruta del archivo CSV donde se guardará la matriz.
    """
    df = pd.DataFrame(cooccurrence_matrix, index=class_names, columns=class_names)
    df.to_csv(file_path)

def plot_cooccurrence_matrix(cooccurrence_matrix, class_names, save_path=None):
    """
    Visualiza la matriz de coocurrencia y opcionalmente la guarda como imagen.

    Args:
        cooccurrence_matrix (np.ndarray): Matriz de coocurrencia.
        class_names (list): Lista de nombres de las clases.
        save_path (str, optional): Ruta del archivo donde se guardará la imagen. 
        Si es None, solo muestra la matriz.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix, annot=True, fmt="d", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Clase')
    plt.ylabel('Clase')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Matriz de Coocurrencia', fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Cierra la figura para liberar memoria
    else:
        plt.show()

def generate_area_percentage_statistics(masks_each, class_names):
    """
    Genera estadísticas por imagen mostrando el porcentaje del área cubierta por cada clase.

    Args:
        masks_each (list or np.ndarray or torch.Tensor): Lista de máscaras predichas para cada imagen.
        class_names (list): Lista de nombres de las clases.

    Returns:
        pd.DataFrame: DataFrame con el porcentaje de área cubierta por cada clase en cada imagen.
    """
    statistics = []
    num_classes = len(class_names)

    for idx, masks in enumerate(masks_each):
        # Convertir a NumPy si es un tensor de PyTorch
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        # Verificar el formato de la máscara y obtener dimensiones
        if masks.ndim == 3 and masks.shape[0] == num_classes:
            height, width = masks.shape[1], masks.shape[2]
            total_pixels = height * width
            busbar_pixels = np.sum(masks[0])  # Clase Busbar
            effective_cell_area = total_pixels - busbar_pixels  # Área efectiva de la celda

            class_areas = {}
            for i, class_name in enumerate(class_names):
                class_pixel_count = np.sum(masks[i])
                
                if class_name == "busbar":
                    class_area_percentage = (class_pixel_count / total_pixels) * 100
                else:
                    class_area_percentage = (class_pixel_count / effective_cell_area) * 100
                
                class_areas[class_name] = round(class_area_percentage, 2)
        elif masks.ndim == 2:
            height, width = masks.shape[0], masks.shape[1]
            total_pixels = height * width
            class_pixel_count = np.sum(masks)
            class_area_percentage = (class_pixel_count / total_pixels) * 100
            class_areas = {class_names[0]: round(class_area_percentage, 2)}
        else:
            raise ValueError(f"Formato de máscara inesperado: {masks.shape}")

        class_areas['image_index'] = 'Image' + str(idx)
        statistics.append(class_areas)

    # Crear un DataFrame con las estadísticas
    df_area_statistics = pd.DataFrame(statistics).fillna(0)

    # Renombrar columnas de clases para agregar el símbolo de porcentaje
    new_class_names = {class_name: f"{class_name} [%]" for class_name in class_names}
    df_area_statistics.rename(columns=new_class_names, inplace=True)
    
    # Invertir el orden de las columnas
    df_area_statistics = df_area_statistics.reindex(columns=df_area_statistics.columns[::-1])

    return df_area_statistics


def save_statistics(statistics_df, file_path):
    """
    Guarda las estadísticas en un archivo CSV.

    Args:
        statistics_df (pd.DataFrame): DataFrame con las estadísticas.
        file_path (str): Ruta del archivo CSV donde se guardarán las estadísticas.
    """
    try:
        statistics_df.to_csv(file_path, index=False)
        print(f"Estadísticas guardadas exitosamente en {file_path}")
    except IOError as e:
        print(f"Error al guardar el archivo CSV: {e}")


# Creación de un panel con las imágenes originales y las máscaras predichas
def combine_panel_predictions(images, masks_each, cols=6, alpha=0.6):
    """
    Combina todas las imágenes del panel con sus predicciones en un arreglo de 16 filas y 6 columnas.

    Args:
        images (list or np.ndarray): Lista de imágenes originales.
        masks_each (list): Lista de máscaras predichas para cada imagen.
        cols (int): Número de columnas en el arreglo del panel.
        alpha (float): Opacidad de las máscaras superpuestas.

    Returns:
        PIL.Image: Imagen combinada del panel completo.
    """
    # Calcular el número de filas necesario para mostrar todas las imágenes
    rows = (len(images) + cols - 1) // cols
    # Asumiendo que todas las imágenes tienen el mismo tamaño
    #img_height, img_width = images[0].shape[:2]
    img_height, img_width = masks_each[0][0].shape[:2]
    colors = {
        'dark': (68, 114, 148),
        'crack': (165, 59, 63),
        'busbar': (222, 156, 83)
    }

    # Crear una imagen vacía para el panel completo
    panel_img = Image.new('RGB', (cols * img_width, rows * img_height))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(images):
                # Dibujar la máscara sobre la imagen correspondiente
                img_with_mask = draw_mask(images[index], masks_each[index], alpha=alpha)
                # Pegar la imagen en la posición correspondiente del panel
                panel_img.paste(img_with_mask, (j * img_width, i * img_height))
    
    return panel_img

def generate_crackcell_stats(img_path, ann_path, busbar_num, crack_inx=100, busbar_inx=10):
    """
    Genera estadísticas de una celda solar con grietas y barras colectoras.

    Args:
        img_path (str): Ruta de la imagen de la celda solar.
        ann_path (str): Ruta del archivo de anotaciones de la celda solar.
        busbar_num (int): Número de barras colectoras en la celda.
        crack_inx (int): Índice de la clase de grietas.
        busbar_inx (int): Índice de la clase de barras colectoras.

    Returns:
        dict: Diccionario con las estadísticas de la celda solar
    """
    # Crear una instancia de CrackCell
    crackcell = CrackCell(img_path, ann_path, crack_inx, busbar_inx, busbar_num)
    
    # Extraer las estadísticas
    inactive_area, inactive_prop = crackcell.extract_inactive_area()
    crack_length = crackcell.extract_crack_length()
    brightness = crackcell.extract_brightness(mode='avg_inactive_only')
    
    # Devolver las estadísticas en un diccionario
    return {
        'Imágen': img_path.split('/')[-1],
        'Área Inactiva (%)': inactive_prop * 100,
        'Longitud de Grieta (pixeles)': crack_length,
        'Brillo': brightness
    }

# Funciones para guardar las visualizaciones como imágenes
def save_heatmap_images(masks_each, out_path):
    """
    Guarda los mapas de calor de las máscaras predichas en archivos PNG.

    Args:
        masks_each (list): Lista de máscaras predichas para cada imagen.
        out_path (str): Ruta de la carpeta donde se guardarán las imágenes.
    """
    heatmap_busbar = generate_heatmap(masks_each, class_index=0)
    heatmap_crack = generate_heatmap(masks_each, class_index=1)
    heatmap_dark = generate_heatmap(masks_each, class_index=2)

    plt.imsave(f'{out_path}/heatmap_busbar.png', heatmap_busbar, cmap='viridis')
    plt.imsave(f'{out_path}/heatmap_crack.png', heatmap_crack, cmap='viridis')
    plt.imsave(f'{out_path}/heatmap_dark.png', heatmap_dark, cmap='viridis')

def generate_defect_bar_chart(area_statistics, save_path):
    """
    Genera un gráfico de barras que muestra cuántas celdas presentan uno o más defectos y guarda la imagen.

    Args:
        area_statistics (pd.DataFrame): DataFrame con las estadísticas de área.
        save_path (str): Ruta donde se guardará la imagen del gráfico.
    """
    # Definir condiciones para clasificar las celdas
    conditions = [
        (area_statistics['dark [%]'] > 0) & (area_statistics['crack [%]'] > 0),
        (area_statistics['dark [%]'] > 0) & (area_statistics['crack [%]'] == 0),
        (area_statistics['dark [%]'] == 0) & (area_statistics['crack [%]'] > 0),
        (area_statistics['dark [%]'] == 0) & (area_statistics['crack [%]'] == 0)
    ]

    # Etiquetas correspondientes
    choices = ['Grietas y Zonas Oscuras', 'Zonas Oscuras', 'Grietas', 'Intactas']

    # Crear una nueva columna con las categorías
    area_statistics['estado_celda'] = np.select(conditions, choices, default='Intactas')

    # Contar cuántas celdas caen en cada categoría
    counts = area_statistics['estado_celda'].value_counts()

    # Graficar el resultado
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Número de Celdas por Estado')
    plt.xlabel('Estado de la Celda')
    plt.ylabel('Número de Imágenes')

    # Agregar anotaciones
    for i in range(len(counts)):
        ax.text(i, counts.values[i] + 0.5, str(counts.values[i]), ha='center', va='bottom')

    # Guardar la imagen
    plt.savefig(save_path)
    plt.close()

def generate_average_area_bar_chart(area_statistics, save_path):
    """
    Genera un gráfico de barras que muestra el área promedio ocupada por cada clase y guarda la imagen.

    Args:
        area_statistics (pd.DataFrame): DataFrame con las estadísticas de área.
        save_path (str): Ruta donde se guardará la imagen del gráfico.
    """
    # Calcular el área promedio ocupada por cada clase
    average_areas = area_statistics[['dark [%]', 'crack [%]', 'busbar [%]']].mean()

    # Crear un gráfico de barras
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=average_areas.index, y=average_areas.values, palette='viridis')

    # Añadir etiquetas con el valor promedio dentro de cada barra
    for index, value in enumerate(average_areas.values):
        ax.text(index, value + 0.1, f'{value:.2f}%', ha='center', va='bottom')

    # Títulos y etiquetas
    plt.title('Área Promedio Ocupada por Cada Clase Predicha')
    plt.xlabel('Clase')
    plt.ylabel('Área Promedio [%]')

    # Guardar la imagen
    plt.savefig(save_path)
    plt.close()

def generate_crackcell_stats_bar_chart(crackcell_stats, save_path):
    """
    Genera un gráfico de barras que muestra los valores promedio de cada columna en crackcell_stats y guarda la imagen.

    Args:
        crackcell_stats (pd.DataFrame): DataFrame con las estadísticas de CrackCell.
        save_path (str): Ruta donde se guardará la imagen del gráfico.
    """
    # Calcular los valores promedio de cada columna
    average_stats = crackcell_stats[['Área Inactiva (%)', 'Longitud de Grieta (pixeles)', 'Brillo']].mean()

    # Crear un gráfico de barras
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=average_stats.index, y=average_stats.values, palette='viridis')

    # Añadir etiquetas con el valor promedio dentro de cada barra
    for index, value in enumerate(average_stats.values):
        ax.text(index, value + 0.2, f'{value:.2f}', ha='center', va='bottom')

    # Títulos y etiquetas
    plt.title('Valores Promedio de Estadísticas de CrackCell')
    plt.xlabel('Estadística')
    plt.ylabel('Valor Promedio')

    # Guardar la imagen
    plt.savefig(save_path)
    plt.close()

# Función para crear una tabla en el PDF
def create_table(dataframe):
    """
    Crea una tabla a partir de un DataFrame de pandas.
    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        
    Returns:
        Table: Tabla de reportlab.
    """
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    return table