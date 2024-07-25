# Importar librerías
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import DataParallel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets.vision import VisionDataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import requests
import copy
from unet_model import construct_unet
from pathlib import Path
from PIL import Image
from imutils.paths import list_images
import os

# Importar Model Handler
from pv_vision.nn import ModelHandler

# ### Predicción con un modelo ya entrenado
class myDataset(Dataset):
    # El método de inicialización se llama cuando se crea una instancia de la clase
    def __init__(self, images, transform):
        # Guardar las imágenes y la transformación como atributos de la instancia
        .....
        # Devolver la imagen transformada
        return image
# Carpeta de imágenes
val_imgs_path = 'D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/CellAnotation_no_humanMasks/dataset_cells/images'
#val_imgs_path = 'D:/Documentos/PV_Vision/crack_segmentation/crack_segmentation/val/img'
# Carpeta de anotaciones
val_annotations_path = 'D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/CellAnotation_no_humanMasks/dataset_cells/annotations'
#val_annotations_path = 'D:/Documentos/PV_Vision/crack_segmentation/crack_segmentation/val/ann'
images_val = [cv.imread(file) for file in list_images(val_imgs_path)]           #imágenes de validación
ann_val = [cv.imread(file) for file in list_images(val_annotations_path)]       # anotaciones de validación
# Crear el dataset
imgset = myDataset(images_val, transform)
# Definir el dispositivo y cargar el modelo
weight_path = 'D:/Documentos/PV_Vision/Neural_Network_W/crack_segmentation/unet_oversample_low_final_model_for_paper/model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Dispositivo: {device}')
unet = construct_unet(5)
unet = DataParallel(unet)
# Inicializar el manejador del modelo
modelhandler = ModelHandler(
    model=unet,
    test_dataset=imgset,
    predict_only=True,
    batch_size_val=2,
    device=device,
    save_dir='D:/Documentos/Universidad de Cuenca/Trabajo de Titulación/Predicciones/Modulos/Celdas/output',
    save_name='unet_cell_prediction'
)
# Cargar los pesos del modelo y generar las predicciones
modelhandler.load_model(weight_path)
masks = modelhandler.predict(save=True)


### Entrenamiento de un modelo nuevo
class SolarDataset(VisionDataset):
    """Un conjunto de datos que lee directamente las imágenes y las máscaras desde una carpeta."""
    ....
# Definición de una clase para componer varias transformaciones.
class Compose:
    def __init__(self, transforms):
        """
        transforms: una lista de transformaciones
        """
        self.transforms = transforms

    # Se definió el método para aplicar las transformaciones a la imagen y la máscara.
    def __call__(self, image, target):
        """
        image: imagen de entrada
        target: máscara de entrada
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
# Se definió una clase para redimensionar la imagen y la máscara a un tamaño fijo.
class FixResize:
    # UNet requiere que el tamaño de entrada sea múltiplo de 16
    def __init__(self, size):
        self.size = size

    # Se definió el método para redimensionar la imagen y la máscara.
    def __call__(self, image, target):
        image = F.resize(image, (self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        target = F.resize(target, (self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST)
        return image, target

# Se definió una clase para transformar la imagen y la máscara a tensores.
class ToTensor:
    """Transforma la imagen a tensor. Escala la imagen a [0,1] float32.
    Transforma la máscara a tensor.
    """
# Se definió una clase para transformar la imagen a tensor manteniendo el tipo original.
class PILToTensor:
    """Transforma la imagen a tensor. Mantiene el tipo original."""

# Se definió una clase para normalizar la imagen.
class Normalize:
    ....
# Ruta al directorio que contiene las imágenes y las máscaras.
root = Path(
    '/content/drive/MyDrive/Trabajo de titulación/PV_vision/Entrenamiento')

# Se definen las transformaciones a aplicar a las imágenes y las etiquetas.
transformers = Compose([FixResize(256), ToTensor(), Normalize()])
# Se crean los conjuntos de datos de entrenamiento, validación y prueba.
trainset = SolarDataset(root, image_folder="train/img",
        mask_folder="train/ann", transforms=transformers)

valset = SolarDataset(root, image_folder="val_resize/img",
        mask_folder="val_resize/ann", transforms=transformers)

testset = SolarDataset(root, image_folder="test/img",
        mask_folder="test/ann", transforms=transformers)
unet = construct_unet(5)
unet = DataParallel(unet)
# Se define el dispositivo en el que se ejecutará el modelo.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = CrossEntropyLoss()
optimizer = Adam(unet.parameters(), lr=0.01)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.2)
# Se inicializa el manejador del modelo.
modelhandler = ModelHandler(
    # Se pasa el modelo que se va a entrenar.
    model = unet,
    # Se pasan los conjuntos de datos de entrenamiento, validación y prueba.
    train_dataset=trainset,
    val_dataset=valset,
    test_dataset=testset,
    # Se especifica el tamaño del lote para el entrenamiento y la validación.
    batch_size_train=16,
    batch_size_val=16,
    # Se pasa el programador de la tasa de aprendizaje.
    lr_scheduler=lr_scheduler,
    # Se especifica el número de épocas para el entrenamiento.
    num_epochs=20,
    # Se pasa la función de pérdida y el optimizador.
    criterion=criterion,
    optimizer=optimizer,
    # Se pasa el dispositivo en el que se ejecutará el entrenamiento.
    device=device,
    # Se especifica el directorio donde se guardarán los puntos de control del modelo.
    save_dir='/content/drive/MyDrive/Trabajo de titulación/PV_vision/Entrenamiento/PuntosControl/checkpoints',
    # Se especifica el nombre del archivo de punto de control.
    save_name='unetv5.pt'
)
# Se inicializa el entrenamiento del modelo.
modelhandler.train_model()