import numpy as np
from sklearn.datasets import make_moons # a simple classification dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
y = y.reshape(-1, 1)

# plot the data here!