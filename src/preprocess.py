import cv2

def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    img = preprocess_image(img)
    return img