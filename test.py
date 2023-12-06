import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

# Carrega o dicionário que mapeia os rótulos para os nomes
with open("label_to_name.pkl", "rb") as file:
    label_to_name = pickle.load(file)

# Carrega o modelo treinado
model = load_model("modelo_mais_brabo_do_universo.keras")

# Função para predizer a pessoa na imagem de teste
def predict_person(model, test_image_path):
    preprocessed_image = preprocess_image(test_image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(preprocessed_image)
    predicted_label = np.argmax(predictions)
    predicted_person = label_to_name[predicted_label]
    return predicted_person

# test_image_path = "./test/Luiz_Marcondes_0001_0000.jpg"
test_image_path = "./test/Erick_test.jpg"
predicted_person = predict_person(model, test_image_path)
print("Pessoa deduzida:", predicted_person)
