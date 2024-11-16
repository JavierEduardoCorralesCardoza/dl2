import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import matplotlib.pyplot as plt

# Cargar el modelo desde el archivo
modelo = load_model('MejorModelo.h5')

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(path, target_size=(128, 128)):
    img = load_img(path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalizar al rango [0,1]
    return img_array

# Diccionario de clases
dict_clases = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

# Directorio de imágenes
directorio = 'test/'

# Listar todos los archivos en el directorio
archivos = [f for f in os.listdir(directorio) if os.path.isfile(os.path.join(directorio, f))]

# Definir la figura y los ejes para el gráfico
num_images = len(archivos)
cols = 3  # Número de columnas en la cuadrícula
rows = (num_images // cols) + (num_images % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))

# Predecir para cada archivo en el directorio y mostrar en la cuadrícula
for i, archivo in enumerate(archivos):
    ax = axes[i // cols, i % cols]
    img_path = os.path.join(directorio, archivo)
    
    # Cargar y preprocesar la imagen
    imagen = np.array([load_and_preprocess_image(img_path)])
    
    # Hacer la predicción
    predicciones = modelo.predict(imagen)
    clase_predicha = np.argmax(predicciones, axis=1)

    # Mostrar la imagen y la predicción
    ax.imshow(imagen[0])
    ax.set_title(f'Predicción: {dict_clases[clase_predicha[0]]}')
    ax.axis('off')

# Si hay espacios vacíos en la cuadrícula, desactivar ejes
for i in range(num_images, rows * cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()
