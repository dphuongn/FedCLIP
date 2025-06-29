import json
import matplotlib.pyplot as plt
import numpy as np

dataset = 'pets'
dataset = 'flowers'
dataset = 'aircraft'
dataset = 'dtd'

partition = 'dir001'

dot_scale = 8  # Try 8, 10, 15, etc. until you like the look

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

plt.figure(figsize=(10, 8))

# Create meshgrid
x, y = np.meshgrid(np.arange(num_clients), np.arange(num_classes))
x, y = x.flatten(), y.flatten()

# Train data
s_train = train_matrix.flatten()
mask_train = s_train > 0
plt.scatter(
    x[mask_train], y[mask_train],
    s=s_train[mask_train]*dot_scale,
    color='red', alpha=0.7, label='Train', edgecolors='k'
)

# Test data
s_test = test_matrix.flatten()
mask_test = s_test > 0
plt.scatter(
    x[mask_test], y[mask_test],
    s=s_test[mask_test]*dot_scale,
    color='blue', alpha=0.7, label='Test', edgecolors='k'
)

plt.xlabel("Client IDs", fontsize=14)
plt.ylabel("Class IDs", fontsize=14)
plt.title("Combined Train & Test Data Distribution", fontsize=16)
plt.xticks(np.arange(num_clients))
plt.yticks(np.arange(0, num_classes, 2))
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{dataset}_{partition}_together.pdf")
plt.show()
