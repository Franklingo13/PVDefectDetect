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