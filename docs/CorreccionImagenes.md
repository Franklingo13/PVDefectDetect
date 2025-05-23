# CorreccionImagenes.py

## Descripción

Este script proporciona una interfaz gráfica de usuario (GUI) para aplicar un flujo de procesamiento a un conjunto de imágenes de electroluminiscencia (EL) de módulos fotovoltaicos. Permite seleccionar carpetas de imágenes, imágenes de fondo y una carpeta de salida, así como definir el nombre de la carpeta donde se guardarán los resultados. El procesamiento incluye sustracción de fondo, corrección de artefactos, cálculo de imágenes promedio/máxima/mínima y mejora de contraste (CLAHE).

## Funcionalidades

- **Selección de carpetas**: Permite seleccionar la carpeta de imágenes EL, la carpeta de imágenes de fondo (BG) y la carpeta de salida mediante cuadros de diálogo.
- **Procesamiento automático**: Al presionar "Enviar", se ejecuta el procesamiento sobre todas las imágenes del dataset.
- **Procesos aplicados**:
  - Sustracción de fondo.
  - Corrección de artefactos.
  - Cálculo de imágenes promedio, máxima y mínima antes y después de CLAHE.
  - Aplicación de CLAHE (mejora de contraste).
- **Visualización**: Muestra imágenes intermedias y finales durante el procesamiento.
- **Guardado de resultados**: Guarda las imágenes procesadas en la carpeta de salida especificada.

## Uso

1. Ejecuta el script.
2. Selecciona la carpeta de imágenes EL, la carpeta de imágenes de fondo y la carpeta de salida.
3. Escribe el nombre de la nueva carpeta donde se guardarán los resultados.
4. Haz clic en "Enviar" para iniciar el procesamiento.

## Dependencias

- tkinter
- matplotlib
- os
- ImagePreprocessing.utils
- ImagePreprocessing.contrast_enhancement

## Estructura del script

- **select_folder()**: Abre un cuadro de diálogo para seleccionar una carpeta.
- **submit()**: Valida las entradas y llama a la función principal de procesamiento.
- **process_images()**: Realiza todo el flujo de procesamiento sobre el dataset.
- **Interfaz gráfica**: Construida con tkinter para facilitar la selección de rutas y parámetros.

## Resultados

Las imágenes procesadas se guardan en la carpeta de salida bajo el nombre de la nueva carpeta especificada, incluyendo:
- Imagen promedio antes y después de CLAHE.
- Imagen máxima y mínima.
- Ejemplos de imágenes intermedias del proceso.