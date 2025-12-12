import numpy as np
from sklearn.datasets import make_moons # simple datatset for training the model
import matplotlib.pyplot as plt
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