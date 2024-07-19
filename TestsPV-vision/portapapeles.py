import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from pv_vision.nn.models import construct_unet
from torch.nn import DataParallel
from imutils.paths import list_images


# Función para convertir las anotaciones de validación en máscaras booleanas de la misma manera que `get_masks` procesa las predicciones del modelo.
def get_annotation_masks(annotations):
    masks_each = []
    for annotation in annotations:
        annotation = cv.resize(annotation, (256, 256))  # Redimensionar anotación al tamaño de las predicciones
        busbar = (annotation == 1)
        crack = (annotation == 2)
        cross = (annotation == 3)
        dark = (annotation == 4)
        # Convertir las máscaras de NumPy a tensores de PyTorch y apilarlas.
        masks_tensor = torch.stack([torch.from_numpy(busbar.astype(np.float32)), 
                                    torch.from_numpy(crack.astype(np.float32)), 
                                    torch.from_numpy(cross.astype(np.float32)), 
                                    torch.from_numpy(dark.astype(np.float32))])
        masks_each.append(masks_tensor)
    return torch.stack(masks_each)

# Definir una función para comparar las máscaras predichas y las máscaras verdaderas
def show_comparison(pred_mask, true_mask, class_names):
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 2, figsize=(10, num_classes * 5))
    
    for i, class_name in enumerate(class_names):
        # Predicted mask for the class
        axes[i, 0].imshow(pred_mask[i], cmap='gray')
        axes[i, 0].set_title(f'Predicted Mask - {class_name}')
        axes[i, 0].axis('off')
        
        # True mask for the class
        axes[i, 1].imshow(true_mask[i], cmap='gray')
        axes[i, 1].set_title(f'True Mask - {class_name}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Definir la transformación de las imágenes que se pasará al manejador del modelo
transform = transforms.Compose([
    # Convertir la imagen a un tensor de PyTorch y escalar los valores de los píxeles entre 0 y 1
    transforms.ToTensor(),
    # Normalizar cada canal de color de la imagen. Los valores de la media y la desviación estándar se especifican para cada canal (RGB). 
    # Estos valores son los valores de media y desviación estándar del conjunto de datos ImageNet.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definir una clase personalizada que hereda de Dataset
class myDataset(Dataset):
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

# Lista de imágenes de validación y anotaciones
val_imgs_path = 'D:/Documentos/PV_Vision/crack_segmentation/crack_segmentation/val/img'
val_annotations_path = 'D:/Documentos/PV_Vision/crack_segmentation/crack_segmentation/val/ann'
images_val = [cv.imread(file) for file in list_images(val_imgs_path)]           # Imágenes de validación
ann_val = [np.array(Image.open(file).convert("L")) for file in list_images(val_annotations_path)]  # Anotaciones de validación en escala de grises

# Verificar que las imágenes se cargaron correctamente
print(f'Número de imágenes de validación: {len(images_val)}')
print(f'Número de anotaciones de validación: {len(ann_val)}')

# Crear el conjunto de datos de imágenes de validación
imgset = myDataset(images_val, transform)

# Definir el dispositivo y cargar el modelo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Dispositivo: {device}')
unet = construct_unet(5)
unet = DataParallel(unet)
weight_path = 'D:/Documentos/PV_Vision/Neural_Network_W/crack_segmentation/unet_oversample_low_final_model_for_paper/model.pt'
unet.load_state_dict(torch.load(weight_path))
unet.eval()    # Establecer el modelo en modo de evaluación

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

# Cargar los pesos del modelo
modelhandler.load_model(weight_path)

# Ejecutar el modelo en el conjunto de datos de prueba
masks = modelhandler.predict(save=True)

# Definir una transformación para redimensionar las imágenes a 256x256 y convertirlas a tensores de PyTorch
resize = transforms.Compose([transforms.Resize((256, 256)), transforms.PILToTensor()])

# Definir un mapa de colores para las diferentes clases de máscaras
color_map = {
    'dark': (68, 114, 148),
    'cross': (77, 137, 99),
    'crack': (165, 59, 63),
    'busbar': (222, 156, 83)
}

# Función para obtener las máscaras de las predicciones del modelo
def get_masks(masks_raw):
    masks_each = []
    masks_all = torch.nn.functional.softmax(torch.from_numpy(masks_raw), dim=1).argmax(dim=1)
    for masks in masks_all:
        busbar = masks == 1
        crack = masks == 2
        cross = masks == 3
        dark = masks == 4
        masks_each.append(torch.dstack([busbar, crack, cross, dark]).permute(2, 0, 1))
    return masks_each

# Función para dibujar las máscaras sobre las imágenes
def draw_mask(img, masks, colors=color_map, alpha=0.6):
    img = Image.fromarray(img)
    img = resize(img)
    combo = draw_segmentation_masks(img, masks, alpha=alpha, colors=[colors[key] for key in ['busbar', 'crack', 'cross', 'dark']])
    return F.to_pil_image(combo)

# Obtener las máscaras de las predicciones del modelo
masks_each = get_masks(masks)

# Obtener las máscaras de las anotaciones de validación
true_masks = get_annotation_masks(ann_val)

# Comparar una máscara predicha con una máscara verdadera
pred_mask = masks_each[0]  # Primera predicción del modelo
true_mask = true_masks[0]  # Primera máscara de anotación

# Nombres de las clases
class_names = ['busbar', 'crack', 'cross', 'dark']

# Mostrar la comparación
show_comparison(pred_mask, true_mask, class_names)
