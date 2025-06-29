import json
import matplotlib.pyplot as plt
import numpy as np

dataset = 'pets'
dataset = 'flowers'
dataset = 'aircraft'
dataset = 'dtd'

partition = 'dir'
# partition = 'dir001'

nc='20'

dot_scale = 8  # Try 8, 10, 15, etc. until you like the look

fontsize = 24 
fontsize_legend = 16
fontsize_small = 20
markersize = 8

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

# Build matrices and combine
train_matrix = build_matrix(client_details, 'train_label_distribution')
test_matrix = build_matrix(client_details, 'test_label_distribution')
combined_matrix = train_matrix + test_matrix

plt.figure(figsize=(10, 8))

# Prepare scatter data
x, y = np.meshgrid(np.arange(num_clients), np.arange(num_classes))
x, y = x.flatten(), y.flatten()
s_combined = combined_matrix.flatten()
mask_combined = s_combined > 0

dot_scale = 8  # Increase for larger dots

# Set y-tick step
c = 15  # <-- Change this value to 20, 40, etc. as desired
y_ticks = np.arange(0, num_classes, c)
if (num_classes - 1) not in y_ticks:
    y_ticks = np.append(y_ticks, num_classes - 1)

plt.scatter(
    x[mask_combined], y[mask_combined],
    s=s_combined[mask_combined]*dot_scale,
    color='red', alpha=0.7, edgecolors='k'
)

plt.xlabel("Client IDs", fontsize=fontsize)
plt.ylabel("Class IDs", fontsize=fontsize)
# plt.title("Combined Train & Test Data Distribution", fontsize=16)
plt.xticks(np.arange(num_clients), fontsize=fontsize)
plt.yticks(y_ticks, fontsize=fontsize)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{dataset}_{partition}_{nc}.pdf")
plt.show()
