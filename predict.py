import numpy as np
from PIL import Image, ImageOps
from network import Network


network = Network(784, 16, 16, 10)
network.load()


def process_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    arr = np.array(img).astype(np.float32)
    arr /= 255.0
    arr = arr.reshape(784)

    return arr


def predict_image(path):
    x = process_image(path)

    prediction = network.predict(x)

    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction: {digit}")


while True:
    path = input("Enter image path (or 'quit'): ")
    if path.lower() == "quit":
        break

    try:
        predict_image(path)

    except Exception as e:
        print("Error:", e)
