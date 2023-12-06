### Alunos
- Erick Garcia
- Jairo Bernardo
- Luiz Marcondes
- Nicolas Souza

### 1. `main.py`

#### Importações:
```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import preprocess_image
import pickle
```

- Importa bibliotecas necessárias, como `os`, `numpy`, `tensorflow`, `layers`, `models`, `preprocess_image` do arquivo `preprocess`, e `pickle`.

#### Carregamento do Dataset:
```python
def load_dataset(dataset_path):
    # Inicialização de listas para imagens e rótulos
    images = []
    labels = []
    label_to_name = {}

    # Itera sobre as pastas no diretório do dataset
    for label, person_name in enumerate(os.listdir(dataset_path)):
        label_to_name[label] = person_name
        person_path = os.path.join(dataset_path, person_name)

        # Itera sobre os arquivos de imagem dentro de cada pasta
        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), label_to_name
```

- A função `load_dataset` carrega as imagens do dataset, pré-processa cada imagem usando a função `preprocess_image` e atribui rótulos a cada imagem. Retorna as imagens, rótulos e um dicionário que mapeia os rótulos para os nomes.

#### Carregamento e Salvamento do Dicionário:
```python
dataset_path = "./post-processed"
images, labels, label_to_name = load_dataset(dataset_path)

# Salva o dicionário que mapeia os rótulos para os nomes
with open("label_to_name.pkl", "wb") as file:
    pickle.dump(label_to_name, file)
```

- Carrega o dataset usando a função `load_dataset`.
- Salva o dicionário `label_to_name` em um arquivo chamado `label_to_name.pkl`.

#### Divisão do Dataset:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
```

- Divide o dataset em conjuntos de treinamento e teste usando `train_test_split` do scikit-learn.

#### Definição da Arquitetura da Rede Neural:
```python
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
```

- Define a arquitetura da rede neural usando o Keras. A arquitetura consiste em camadas convolucionais, de pooling, flatten, densa e dropout.

#### Compilação e Treinamento do Modelo:
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))
```

- Compila o modelo usando o otimizador 'adam' e a função de perda 'sparse_categorical_crossentropy'.
- Treina o modelo com os conjuntos de treinamento e teste.

#### Salvamento do Modelo Treinado:
```python
model.save("modelo_mais_brabo_do_universo.keras")
```

- Salva o modelo treinado em um arquivo chamado `modelo_mais_brabo_do_universo.keras`.

### 2. `test.py`

#### Carregamento do Dicionário e Modelo Treinado:
```python
with open("label_to_name.pkl", "rb") as file:
    label_to_name = pickle.load(file)

model = load_model("modelo_mais_brabo_do_universo.keras")
```

- Carrega o dicionário `label_to_name` e o modelo treinado.

#### Função para Predizer a Pessoa na Imagem de Teste:
```python
def predict_person(model, test_image_path):
    preprocessed_image = preprocess_image(test_image_path)
    predictions = model.predict(preprocessed_image)
    predicted_label = np.argmax(predictions)
    predicted_person = label_to_name[predicted_label]
    return predicted_person
```

- Define uma função `predict_person` que recebe um modelo e o caminho da imagem de teste.
- Pré-processa a imagem, faz predições usando o modelo e retorna a pessoa prevista.

#### Teste com uma Imagem:
```python
test_image_path = "./test/Luiz_Marcondes_0001_0000.jpg"
predicted_person = predict_person(model, test_image_path)
print("Pessoa deduzida:", predicted_person)
```

- Testa a função `predict_person` com uma imagem específica e imprime o resultado.

### 3. `preprocess.py`

#### Pré-processamento da Imagem de Teste:
```python
import cv2

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)) 
    img = img / 255.0
    return img
```

- Define a função `preprocess_image` que carrega uma imagem, converte para o formato RGB, redimensiona para (112, 112) e normaliza os valores dos pixels para o intervalo [0, 1].
