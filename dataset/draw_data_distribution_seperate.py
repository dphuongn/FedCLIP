import json
import matplotlib.pyplot as plt
import numpy as np

dataset = 'pets'

partition = 'dir001'

# Path to your JSON file
json_path = f"{dataset}/config.json"

# Load JSON data
with open(json_path, "r") as f:
    data = json.load(f)

client_details = data["client_details"]
num_clients = data["num_clients"]
num_classes = data["num_classes"]

def build_matrix(client_details, key):
    matrix = np.zeros((num_classes, num_clients))
    for client in client_details:
        client_id = client['client_id']
        dist = client[key]
        for cls, count in dist.items():
            matrix[int(cls), client_id] = count
    return matrix

# Build matrices for train and test
train_matrix = build_matrix(client_details, 'train_label_distribution')
test_matrix = build_matrix(client_details, 'test_label_distribution')

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
titles = ['Train Data Distribution', 'Test Data Distribution']

for i, matrix in enumerate([train_matrix, test_matrix]):
    ax = axes[i]
    x, y = np.meshgrid(np.arange(num_clients), np.arange(num_classes))
    x, y, s = x.flatten(), y.flatten(), matrix.flatten()
    nonzero = s > 0
    sc = ax.scatter(
        x[nonzero], y[nonzero],
        s=s[nonzero] * 1.5,
        alpha=0.8, color='red', edgecolors='k'
    )
    ax.set_xlabel("Client IDs", fontsize=14)
    ax.set_ylabel("Class IDs" if i == 0 else "", fontsize=14)
    ax.set_title(titles[i], fontsize=16)
    ax.set_xticks(np.arange(num_clients))
    ax.set_yticks(np.arange(0, num_classes, 2))
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{dataset}_{partition}_seperate.pdf")  # Save as PDF
plt.show()
