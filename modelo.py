import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
dataset = pd.read_csv('data.csv')

# Verifica que el archivo se haya cargado correctamente
print(dataset.head())

def load_and_preprocess_image(path, target_size=(32, 32)):
    path = "dataset/" + path
    img = load_img(path, target_size=target_size)
    print(path)
    img_array = img_to_array(img) / 255.0  # Normalizar al rango [0,1]
    return img_array


# Convertir etiquetas a categorías
sentimientos = dataset['label'].astype('category').cat.codes
labels = to_categorical(sentimientos)

# Convertir las etiquetas a categorías y obtener las clases
clases = dataset['label'].astype('category').cat.categories
clases_dict = dict(enumerate(clases))

# Imprimir el diccionario de clases
print(clases_dict)


# Cargar y preprocesar las imágenes
images = np.array([load_and_preprocess_image(path) for path in dataset['path']])
np.save('imagenes_procesadas.npy', images)

images = np.load('imagenes_procesadas.npy')
labels = np.array(labels)

# Dividir en datos de entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=42)

# Configuración de Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalización de las imágenes
    rotation_range=30,  # Rotación aleatoria de hasta 30 grados
    width_shift_range=0.2,  # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,  # Desplazamiento vertical aleatorio
    shear_range=0.2,  # Cizallamiento
    zoom_range=0.2,  # Zoom aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    fill_mode='nearest'  # Rellenar los píxeles faltantes
)

# El generador para validación no necesita augmentación, solo normalización
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Fit el generador de datos de entrenamiento
train_generator = train_datagen.flow(x_train, y_train, batch_size=64)
val_generator = val_datagen.flow(x_val, y_val, batch_size=64)

model = Sequential([
    Input(shape=(64, 64, 3)),  # Capa de entrada explícita
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(labels.shape[1], activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar el EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',  # Monitorea la pérdida en los datos de validación
                               patience=5,          # Número de épocas sin mejora antes de detener el entrenamiento
                               verbose=1,           # Muestra información sobre el early stopping
                               restore_best_weights=True)  # Restaura los pesos del modelo cuando se detiene

# Entrenar el modelo con EarlyStopping
history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=100, 
                    validation_data=(x_val, y_val), 
                    verbose=1,
                    callbacks=[early_stopping])

# Evaluar el modelo en los datos de validación
loss, accuracy = model.evaluate(x_val, y_val)
print(f'Pérdida en los datos de validación: {loss}')
print(f'Precisión en los datos de validación: {accuracy}')

# Realizar predicciones sobre los datos de validación
y_pred = model.predict(x_val)

# Convertir las predicciones a etiquetas de clase
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

# Calcular el F1 score
f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')  # 'weighted' para ponderar por el tamaño de cada clase
print(f'F1 Score: {f1}')

# Graficar las pérdidas de entrenamiento y validación
plt.plot(history.history['loss'], label='Loss de entrenamiento')
plt.plot(history.history['val_loss'], label='Loss de validación')

# Agregar etiquetas y título
plt.title('Loss de entrenamiento vs Loss de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Mostrar el gráfico
plt.show()

model.save('MejorModelo.keras')
