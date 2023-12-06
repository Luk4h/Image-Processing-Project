import cv2

# Pre processamento da imagem de teste
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112)) 
    img = img / 255.0
    return img