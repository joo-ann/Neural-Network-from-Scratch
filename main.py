import numpy as np
from network import Network
from torchvision import datasets, transforms

if __name__ == "__main__":
    network = Network(784, 16, 16, 10)
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    inputs = []
    labels = []

    for i, j in train_dataset:
        x = i.view(784).numpy()
        y = np.zeros(10)
        y[j] = 1
        
        inputs.append(x)
        labels.append(y)


    inputs = np.array(inputs)
    labels = np.array(labels)

    network.train(inputs[:5000], labels[:5000], epochs=100)
