<div align="center" id="top"> 
  <img src="./.github/app.gif" alt="PVDefectDetect" />

  &#xa0;

  <!-- <a href="https://pvdefectdetect.netlify.app">Demo</a> -->
</div>

<h1 align="center">PVDefectDetect</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/Franklingo13/PVDefectDetect?color=56BEB8">

  <img alt="Github language count" src="https://img.shields.io/github/languages/count/Franklingo13/PVDefectDetect?color=56BEB8">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/Franklingo13/PVDefectDetect?color=56BEB8">

  <img alt="License" src="https://img.shields.io/github/license/Franklingo13/PVDefectDetect?color=56BEB8">

  <!-- <img alt="Github issues" src="https://img.shields.io/github/issues/Franklingo13/PVDefectDetect?color=56BEB8" /> -->

  <!-- <img alt="Github forks" src="https://img.shields.io/github/forks/Franklingo13/PVDefectDetect?color=56BEB8" /> -->

  <!-- <img alt="Github stars" src="https://img.shields.io/github/stars/Franklingo13/PVDefectDetect?color=56BEB8" /> -->
</p>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  PVDefectDetect 🚀 Bajo construcción...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-tecnologías">Tecnologías</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requisitos">Requisitos</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-comenzando">Comenzando</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/{{YOUR_GITHUB_USERNAME}}" target="_blank">Author</a>
</p>

<br>

## :dart: About ##


Este archivo README proporciona una descripción general de la estructura y el contenido de tu proyecto.

**Estructura de Carpetas:**

* **checkpoints** (Puestos de Control): Esta carpeta almacena modelos entrenados en diferentes etapas. 
* **EjemplosPV-vision** (Ejemplos PV-Vision): Contiene archivos de ejemplo para probar los tutoriales de Jupyter Notebook relacionados con la visión artificial para paneles solares (PV-Vision).
    * **module_imgs** (Imágenes de Módulo): Contiene imágenes originales de electroluminiscencia (EL) de módulos solares provenientes de campo y laboratorio.
    * **raw_img_gray** (Imágenes Crudas en Escala de Grises):  **(Aclaración necesaria)** Se desconoce el contenido exacto de esta carpeta debido a un Acuerdo de No Divulgación (NDA).
* **ImagenesXcap** (Imágenes Xcap): Almacena un conjunto de imágenes EL tomadas con una cámara SWIR editadas con XCAP.
    * **070A_8v**: Probablemente, un identificador específico para este conjunto de imágenes.
* **Test_crack** (Pruebas de Grietas): Contiene 10 imágenes del conjunto de datos que presentan grietas y fracturas en paneles solares.
* **Test_ImageJ** (Pruebas de ImageJ): Contiene imágenes EL editadas con el software ImageJ.
    * **stack_070_edited**: Probablemente, un nombre específico para esta imagen editada.
* **TestsPV-vision** (Pruebas PV-Vision): Contiene cuadernos Jupyter Notebook para seguir los tutoriales de la librería PV-Vision para el análisis de imágenes de paneles solares.
    * **checkpoints** (Puestos de Control): Almacena modelos entrenados en diferentes épocas (por ejemplo, `epoch_10`). 
    * **examples** (Ejemplos): Contiene ejemplos para utilizar las funcionalidades de la librería PV-Vision.
        * **cell_classification** (Clasificación de Celdas): Contiene imágenes de celdas solares individuales recortadas de módulos completos. Estas celdas se clasifican según las etiquetas manuales proporcionadas en la carpeta `../object_detection/yolo_manual_ann`. 
        * **crack_segmentation** (Segmentación de Grietas): 
            * **img_for_crack_analysis** (Imágenes para Análisis de Grietas): **(Aclaración necesaria)** Se desconoce el propósito exacto de esta subcarpeta.
            * **img_for_prediction** (Imágenes para Predicción): **(Aclaración necesaria)** Se desconoce el propósito exacto de esta subcarpeta.
            * **img_label_for_training** (Imágenes con Etiquetas para Entrenamiento): Contiene imágenes segmentadas que sirven para entrenar modelos de detección de grietas. 
                * **testset** (Conjunto de Prueba): 
                    * **ann** (Anotaciones): Contiene información sobre las grietas presentes en las imágenes de prueba.
                    * **img** (Imágenes): Contiene las imágenes del conjunto de prueba.
                * **train** (Conjunto de Entrenamiento): 
                    * **ann** (Anotaciones): Contiene información sobre las grietas presentes en las imágenes de entrenamiento.
                    * **img** (Imágenes): Contiene las imágenes del conjunto de entrenamiento.
                * **val** (Conjunto de Validación): 
                    * **ann** (Anotaciones): Contiene información sobre las grietas presentes en las imágenes de validación.
                    * **img** (Imágenes): Contiene las imágenes del conjunto de validación.
        * **object_detection** (Detección de Objetos): Contiene imágenes de módulos solares transformadas en perspectiva para facilitar la detección de celdas defectuosas. Además, la subcarpeta `yolo_manual_ann` almacena anotaciones manuales que indican la posición de dichas celdas.
        * **transform_seg** (Segmentación con Transformación): 
            * **field_pipeline** (Flujo de Campo): Contiene imágenes EL originales de campo en formato RGB o escala de grises. Las anotaciones para la transformación de perspectiva se encuentran en la subcarpeta `unet_ann`. El mapa de colores para transformar imágenes RGB a escala de grises no se incluye debido al NDA.
            * **module_imgs** (Imágenes de Módulo): Contiene imágenes EL originales provenientes de campo y laboratorio.

**Recomendación:**

Se recomienda utilizar la carpeta `module_imgs` para practicar con las herramientas de transformación de módulos y segmentación de celdas. Los datos de `field_pipeline` se utilizan en tutoriales que manejan un gran número de imágenes de campo.



## :sparkles: Features ##

:heavy_check_mark: Feature 1;\
:heavy_check_mark: Feature 2;\
:heavy_check_mark: Feature 3;

## :rocket: Tecnologías ##

Las siguientes herramientas fueron utilizadas en este proyecto:

- [Python](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## :white_check_mark: Requisitos ##

Antes de comenzar :checkered_flag:, necesitas tener [Git](https://git-scm.com) y [Python](https://www.python.org/) instalados.

## :checkered_flag: Comenzando ##

```bash
# Clona este proyecto
$ git clone https://github.com/Franklingo13/PVDefectDetect.git

# Accede al directorio del proyecto
$ cd pvdefectdetect

# Instala las dependencias
$ pip install -r requirements.txt

# Corre un script de ejemplo
$ python AnalizarMetricas.py
```

## :memo: License ##

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo [LICENSE](https://github.com/Franklingo13/PVDefectDetect/blob/839f48570a84f0ec2ab96b3bccbd602e0f8ebfb5/LICENSE) file.

 
Realizado por [Franklingo13](https://github.com/Franklingo13)

&#xa0;

<a href="#top">Back to top</a>
