import numpy as np
from network import Network
from torchvision import datasets, transforms

if __name__ == "__main__":
    # initialize network with 784 inputs (every pixel in a 28x28), 2 hidden layers of 16 nodes, and an output layer of 10 nodes (0-9)
    network = Network(784, 16, 16, 10)
    transform = transforms.ToTensor()

    # importing the dataset
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

    # tuning the dataset for viable input into the network
    for i, j in train_dataset:
        x = i.view(784).numpy()
        y = np.zeros(10)
        y[j] = 1
        
        inputs.append(x)
        labels.append(y)


    inputs = np.array(inputs)
    labels = np.array(labels)

    # training :)
    network.train(inputs[:5000], labels[:5000], epochs=100)

'''
Final Training Results

Epoch 100 has a training loss of 3.88% and a validation loss of 3.71%
'''
