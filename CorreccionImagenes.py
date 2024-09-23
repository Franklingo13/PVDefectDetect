import tkinter as tk
from tkinter import filedialog, messagebox
import os
import matplotlib.pyplot as plt
#import pandas as pd
#import numpy as np
from ImagePreprocessing.utils import *
from ImagePreprocessing.contrast_enhancement import *
#from EvaluationMetrics.evaluationMetrics import *

# Aplicar diferentes correcciones a un dataset de imágenes

# Directorio de las imágenes
# path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino\Poli2_Inf_V40_I4.5_T28\EL"  
# bg_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino\Poli2_Inf_V40_I4.5_T28\Fondo"
# out_path = r"D:\Documentos\Universidad de Cuenca\Trabajo de Titulación\Datasets_EL\Policristalino"
# nombre_carpeta = "Poli2_Inf_ImagenesCorregidas"

def select_folder():
    folder_selected = filedialog.askdirectory()
    return folder_selected

def submit():
    global path, bg_path, out_path, nombre_carpeta
    path = entry_imgs_path.get()
    bg_path = entry_bg_path.get()
    out_path = entry_out_path.get()
    nombre_carpeta = entry_nombre_carpeta.get()
    
    # Validar las entradas
    if not path or not bg_path or not out_path or not nombre_carpeta:
        messagebox.showerror("Error", "Todos los campos son obligatorios")
        return
    
    root.destroy()
    process_images(path, bg_path, out_path, nombre_carpeta)

def process_images(path, bg_path, out_path, nombre_carpeta):
    salidas_path = read_folder_path(out_path)
    dataset_path = read_folder_path(path)
    BG_dataset_path = read_folder_path(bg_path)
    print("Directorio imagenes: ", dataset_path)
    print("Directorio fondo: ", BG_dataset_path)

    dataset = read_images(dataset_path)
    datasetBG = read_images(BG_dataset_path)
    print("Número de imágenes en el dataset:", len(dataset))

    # muestra la imagen dataset[0]
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

    # Aplicar el algoritmo de mejora de contraste (CLAHE) a las imágenes sin fondo
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
    os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True)
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "mean_image.jpg"), mean_image)
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "max_image.jpg"), max_image)
    cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "min_image.jpg"), min_image)

root = tk.Tk()
root.title("Parámetros del Análisis de Imágenes EL")

# Carpeta de imágenes EL
tk.Label(root, text="Carpeta de Imágenes EL:").grid(row=0, column=0, padx=10, pady=5)
entry_imgs_path = tk.Entry(root, width=50)
entry_imgs_path.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_imgs_path.insert(0, select_folder())).grid(row=0, column=2, padx=10, pady=5)

# Carpeta de imágenes de fondo (BG)
tk.Label(root, text="Carpeta de Imágenes de Fondo (BG):").grid(row=1, column=0, padx=10, pady=5)
entry_bg_path = tk.Entry(root, width=50)
entry_bg_path.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_bg_path.insert(0, select_folder())).grid(row=1, column=2, padx=10, pady=5)

# Carpeta de salida
tk.Label(root, text="Carpeta de Salida:").grid(row=2, column=0, padx=10, pady=5)
entry_out_path = tk.Entry(root, width=50)
entry_out_path.grid(row=2, column=1, padx=10, pady=5)
tk.Button(root, text="Seleccionar", command=lambda: entry_out_path.insert(0, select_folder())).grid(row=2, column=2, padx=10, pady=5)

# Nombre de la nueva carpeta
tk.Label(root, text="Nombre de la Nueva Carpeta:").grid(row=3, column=0, padx=10, pady=5)
entry_nombre_carpeta = tk.Entry(root, width=50)
entry_nombre_carpeta.grid(row=3, column=1, padx=10, pady=5)

# Botón de enviar
tk.Button(root, text="Enviar", command=submit).grid(row=4, column=1, padx=10, pady=20)

root.mainloop()


salidas_path = read_folder_path(out_path) # type: ignore
dataset_path = read_folder_path(path)
BG_dataset_path = read_folder_path(bg_path) # type: ignore
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

# Aplicar el algoritmo de mejora de contraste (CLAHE) a las imágenes sin fondo
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
os.makedirs(os.path.join(salidas_path, nombre_carpeta), exist_ok=True) # type: ignore
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "mean_image.jpg"), mean_image) # type: ignore
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "max_image.jpg"), max_image) # type: ignore
cv2.imwrite(os.path.join(salidas_path, nombre_carpeta, "min_image.jpg"), min_image) # type: ignore

