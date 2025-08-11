<div align="center" id="top">
  <img src="./.github/app.gif" alt="PVDefectDetect" />

  &#xa0;

  </div>

<h1 align="center">PVDefectDetect</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/Franklingo13/PVDefectDetect?color=56BEB8">
  <img alt="Github language count" src="https://img.shields.io/github/languages/count/Franklingo13/PVDefectDetect?color=56BEB8">
  <img alt="Repository size" src="https://img.shields.io/github/repo-size/Franklingo13/PVDefectDetect?color=56BEB8">
  <img alt="License" src="https://img-shields.io/github/license/Franklingo13/PVDefectDetect?color=56BEB8">
  </p>

<p align="center">
  <a href="#dart-acerca-del-proyecto">Acerca del proyecto</a> &#xa0; | &#xa0;
  <a href="#sparkles-características">Características</a> &#xa0; | &#xa0;
  <a href="#rocket-tecnologías">Tecnologías</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requisitos">Requisitos</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-guía-de-uso">Guía de uso</a> &#xa0; | &#xa0;
  <a href="#memo-licencia">Licencia</a> &#xa0; | &#xa0;
  <a href="https://github.com/Franklingo13" target="_blank">Autor</a>
</p>

<br>

## :dart: Acerca del proyecto

Este proyecto se centra en la **detección de defectos en módulos fotovoltaicos** a partir de imágenes de electroluminiscencia (EL). Para ello, combina técnicas avanzadas de preprocesamiento de imágenes con segmentación semántica mediante redes neuronales como **U-Net**.

El flujo de trabajo completo abarca desde la captura inicial de imágenes hasta la evaluación detallada del desempeño del sistema.

### Metodología

- **Adquisición de imágenes EL**: Utilización de cámaras de alta sensibilidad (p. ej., cámara OWL 640 M) para capturar las imágenes.
- **Pipeline de preprocesamiento**: Un flujo de trabajo robusto que incluye sustracción de fondo, remoción de artefactos, realce de contraste (mediante CLAHE) y recorte de imágenes a nivel de celda.
- **Segmentación semántica**: Aplicación de modelos U-Net para la detección y clasificación de defectos comunes, como fisuras (cracks), barras colectoras (busbars) y zonas oscuras (dark).
- **Evaluación del sistema**: Análisis exhaustivo del rendimiento del modelo, tanto a nivel objetivo (métricas, SNR) como subjetivo (visual), utilizando matrices de confusión, mapas de error y mapas de probabilidad (softmax).

![Diagrama de Metodología del proyecto](Diagrama_Metodologia.png)

---

## :sparkles: Características principales

- **Preprocesamiento avanzado**: Scripts para sustracción de fondo, realce de contraste (CLAHE) y remoción de artefactos en imágenes EL.
- **Segmentación a nivel de celda**: Manejo eficiente de regiones de interés (ROI) y corrección de deformaciones para un análisis preciso.
- **Modelos de segmentación**: Implementación y evaluación de modelos U-Net con notebooks dedicados para entrenamiento y métricas.
- **Visualización de resultados**: Herramientas para superponer máscaras, generar mapas de error (FP/FN) y visualizar heatmaps de probabilidad por clase.

---

## :rocket: Tecnologías

- **Lenguajes**: Python
- **Frameworks**: PyTorch, Jupyter Notebook
- **Librerías**: OpenCV, NumPy, Matplotlib, Seaborn

---

## :white_check_mark: Requisitos

- Python 3.10+
- `pip`
- Se recomienda el uso de un entorno virtual (venv).

Para instalar las dependencias, ejecuta los siguientes comandos en tu terminal:

```bash
# En sistemas Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# En sistemas macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

-----

## :checkered\_flag: Guía de uso

Aquí te mostramos cómo interactuar con las funcionalidades principales del repositorio.

### Estructura del proyecto

  - `ImagePreprocessing/`: Contiene scripts y notebooks para el preprocesamiento de imágenes.
  - `Deteccion_de_Grietas_y_Pruebas_de_Modelos/`: Almacena notebooks y modelos para la segmentación y evaluación.
  - `Predicciones/`: Utilidades para la inferencia y visualización.
  - `EvaluationMetrics/`: Scripts para el cálculo de métricas de rendimiento.
  - `SNR_calc/`: Scripts para el cálculo de la relación señal-ruido (SNR) según la norma IEC.
  - `docs/`: Documentación complementaria.

### Flujo de trabajo rápido

1.  **Preprocesamiento**: Utiliza `CorreccionImagenes.py` para procesar un conjunto de imágenes desde una carpeta de entrada y guardar los resultados en una carpeta de salida. También puedes explorar los notebooks en `ImagePreprocessing/` para un análisis más interactivo.

2.  **Entrenamiento y evaluación**: Accede a los notebooks en `Deteccion_de_Grietas_y_Pruebas_de_Modelos/` para entrenar el modelo, generar métricas detalladas, matrices de confusión y mapas de error.

3.  **Inferencia y visualización**: Usa `Predicciones/Predicciones.py` o su notebook asociado para aplicar el modelo a nuevas imágenes. Obtendrás como salida las imágenes con máscaras superpuestas y mapas de probabilidad.

**Nota**: Asegúrate de organizar tus imágenes de electroluminiscencia (y las de fondo) en carpetas claras. Los scripts te pedirán las rutas de entrada para funcionar correctamente.

-----

## :memo: Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Para más detalles, consulta el archivo [LICENSE](https://www.google.com/search?q=LICENSE).

Autor: [Franklingo13](https://github.com/Franklingo13)

<br>

<p align="center"><a href="#top">Volver arriba</a></p>

