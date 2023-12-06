import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import preprocess_image
import pickle

# Função para carregar o dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    label_to_name = {}

    for label, person_name in enumerate(os.listdir(dataset_path)):
        label_to_name[label] = person_name
        person_path = os.path.join(dataset_path, person_name)

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), label_to_name

# Carrega o dataset
dataset_path = "./post-processed"
images, labels, label_to_name = load_dataset(dataset_path)

# Salva o dicionário que mapeia os rótulos para os nomes
with open("label_to_name.pkl", "wb") as file:
    pickle.dump(label_to_name, file)

# Divide o dataset em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define a arquitetura da rede neural
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_to_name), activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Salva o modelo
model.save("modelo_mais_brabo_do_universo.keras")
