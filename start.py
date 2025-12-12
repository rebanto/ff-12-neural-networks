import numpy as np
from sklearn.datasets import make_moons # a simple classification dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
y = y.reshape(-1, 1)

df = pd.DataFrame(X, columns=['X_1', 'X_2'])
df['y'] = y

df_head = df.head(10)

print(df_head.to_markdown(index=False, numalign="left", stralign="left"))

# graphing
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Make Moons Dataset')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.grid(True, linestyle='--', alpha=0.6)

legend1 = plt.legend(*scatter.legend_elements(), title='Classes')
plt.gca().add_artist(legend1)

# Ensure plots directory exists and save the dataset scatter image
os.makedirs('plots', exist_ok=True)
plt.savefig(os.path.join('plots', 'dataset_scatter.png'), bbox_inches='tight', dpi=150)
plt.show()

import random

# class Neuron:
#     def __init__(self, f1, f2):
#         self.f1 = f1
#         self.f2 = f2
#         self.w1 = random.random()
#         self.w2 = random.random()
#         self.b = random.random()

#     def forward(self):
#         return self.w1 * self.f1 + self.w2 * self.f2 + self.b

class MSE:
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        self.dinputs = (2 * self.diff) / samples

class ReLU:
    def forward(self, x):
        self.inputs = x
        self.output = np.maximum(0, x)

    def backward(self, d_out):
        self.dinputs = d_out * (self.inputs > 0)

class Sigmoid:
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))

    def backward(self, d_out):
        self.dinputs = d_out * self.output * (1 - self.output)

class Dense:
    def __init__(self, in_features, out_features):
        self.weights = 0.1 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))

    def forward(self, x):
        self.inputs = x
        self.output = np.dot(x, self.weights) + self.biases

    def backward(self, d_out):
        self.dweights = np.dot(self.inputs.T, d_out)
        self.dbiases = np.sum(d_out, axis=0, keepdims=True)
        self.dinputs = np.dot(d_out, self.weights.T)

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.bases -= self.lr * layer.dbiases

