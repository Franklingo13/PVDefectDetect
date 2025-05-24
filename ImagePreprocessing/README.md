# Procedimiento de Corrección de Imágenes EL

Este script proporciona un procedimiento para la corrección de imágenes de electroluminiscencia (EL) utilizando una interfaz gráfica de usuario (GUI) para facilitar la selección de parámetros.

## Requisitos

- Python 3.x
- Bibliotecas de Python:
  - `tkinter`
  - `cv2` (OpenCV)
  - `numpy`
  - `matplotlib`

Se puede instalar las bibliotecas necesarias utilizando `pip`:

```sh
pip install opencv-python numpy tkinter matplotlib
```

## Estructura del Proyecto

- `CorreccionImagenes.py`: Script principal que realiza la corrección de imágenes.
- `MMCE.ipynb`: Definición del modelo U-Net.
- `contrast_enhancement.py`: Funciones utilizadas para la corrección de imágenes.
- `utils.py`: Funciones auxiliares utilizadas.
- `EvaluacionSubjetiva.ipynb`: Notebook de Jupyter con
- `el_image_processing.ipynb`: Notebook de Jupyter con 

## Uso

1. Ejecuta el script `CorreccionImagenes.py`:
    ```sh
    python CorreccionImagenes.py
    ```

2. Se abrirá una ventana de la GUI donde se podrá seleccionar las carpetas y parámetros necesarios:
    - **Carpeta de Imágenes EL**: Selecciona la carpeta que contiene las imágenes de electroluminiscencia.
    - **Carpeta de Imágenes de Fondo (BG)**: Selecciona la carpeta que contiene las imágenes de fondo.
    - **Carpeta de Salida**: Selecciona la carpeta donde se guardarán las imágenes corregidas.
    - **Nombre de la Nueva Carpeta**: Ingresa el nombre de la nueva carpeta que se creará para guardar las imágenes corregidas.

3. Haz clic en el botón "Enviar" para iniciar el proceso de corrección de imágenes.

## Créditos

Parte del código de este proyecto emplea algoritmos del repositorio [imgProcessor](https://github.com/radjkarl/imgProcessor).

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más detalles.



# Evaluación Subjetiva de Imágenes Mejoradas

## Descripción

Este notebook permite realizar una **evaluación visual subjetiva** de imágenes de electroluminiscencia (EL) de módulos fotovoltaicos, mostrando cada imagen junto a una región de interés (ROI) ampliada. El objetivo es comparar visualmente el efecto de diferentes etapas de procesamiento (como sustracción de fondo, eliminación de artefactos y transformaciones geométricas) sobre las imágenes originales y procesadas.

## Funcionalidades principales

- **Carga de imágenes originales y procesadas** desde rutas configurables.
- **Selección y visualización de una ROI** (región de interés) en cada imagen, con posibilidad de ajustar su posición y tamaño.
- **Visualización lado a lado** de la imagen completa con el ROI marcado y la zona ampliada.
- **Guardado automático** de las siguientes imágenes para documentación o artículos:
  1. Imagen original con el ROI marcado.
  2. Imagen original ampliada en el ROI.
  3. Imagen transformada con el ROI marcado.
  4. Imagen transformada ampliada en el ROI.
- **Comparación visual** de diferentes etapas de procesamiento sobre un mismo conjunto de imágenes.

## Estructura del notebook

1. **Importación de librerías**: Incluye OpenCV, Matplotlib, Numpy y utilidades propias.
2. **Definición de funciones**:  
   - `mostrar_imagen_con_roi`: Muestra una imagen con el ROI marcado y la zona ampliada, sin deformaciones.
3. **Carga de imágenes**:  
   - Se cargan imágenes originales y transformadas desde rutas configurables.
   - Se verifica la correcta carga de las imágenes.
4. **Selección y ajuste del ROI**:  
   - Se define el tamaño y la posición del ROI, así como posibles desplazamientos para alinear regiones entre imágenes de diferente resolución o alineación.
5. **Visualización y guardado de resultados**:  
   - Se muestran y guardan las imágenes solicitadas en el directorio base.
6. **Comparación de etapas de procesamiento**:  
   - Se visualizan diferentes versiones de la imagen (original, sin fondo, sin artefactos, etc.) con su respectivo ROI.

## Uso recomendado

1. **Configura las rutas** de tus imágenes en las celdas correspondientes.
2. **Ajusta el ROI** (posición y tamaño) según la zona de interés que desees analizar.
3. **Ejecuta las celdas** para visualizar y guardar los resultados.
4. Utiliza las imágenes generadas para documentación, artículos o análisis subjetivo.

## Dependencias

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib
- Numpy
- utils propios del proyecto

## Notas

- El notebook está pensado para ser fácilmente adaptable a otros conjuntos de imágenes o etapas de procesamiento.
- Los nombres de los archivos generados y las rutas de salida pueden modificarse según las necesidades del usuario.
- El ROI puede desplazarse en la imagen transformada para compensar posibles desalineaciones.

---
