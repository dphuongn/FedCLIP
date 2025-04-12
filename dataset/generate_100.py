import numpy as np
from pathlib import Path
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, process_dataset, separate_data, separate_data_pfl, split_data, save_file, separate_data_few_shot_iid, separate_data_few_shot_pat_non_iid

random.seed(1)
np.random.seed(1)

dir_path = Path("aircraft")
num_classes = 100

# Allocate data to users
def generate_aircraft(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        
    # Setup directory for train/test data
    config_path = dir_path / "config.json"
    train_path = dir_path / "train"
    test_path = dir_path / "test"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl):
        return
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize all images to 224x224
        transforms.ToTensor() # Then convert them to tensor
    ])

    trainset = torchvision.datasets.FGVCAircraft(
        root=dir_path / "rawdata", split='trainval', download=True, transform=transform)
    testset = torchvision.datasets.FGVCAircraft(
        root=dir_path / "rawdata", split='test', download=True, transform=transform)
    
    
    # Process the datasets
    train_images, train_labels = process_dataset(trainset)
    test_images, test_labels = process_dataset(testset)

    print(f'train_images length: {len(train_images)}')
    print(f'train_labels length: {len(train_labels)}')
    print(f'test_images length: {len(test_images)}')
    print(f'test_labels length: {len(test_labels)}')

    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Combine train and test datasets separately
    dataset_image = np.concatenate((train_images, test_images), axis=0)
    dataset_label = np.concatenate((train_labels, test_labels), axis=0)
    
    if pfl:
        X, y, statistic = separate_data_pfl((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
        
        train_data, test_data = split_data(X, y)
        
        for idx, test_dict in enumerate(train_data):
            print(f'train data: {idx}')
            print(f'train data shape: {len(train_data[idx]["y"])}')
        for idx, test_dict in enumerate(test_data):
            print(f'test data: {idx}')
            print(f'test data shape: {len(test_dict["x"])}')
    

    elif few_shot:  # Add a parameter or a condition to trigger few-shot scenario
        if not niid:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        else:
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_pat_non_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        
    else:

        train_data, test_data, statistic, statistic_test = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
    
    
    # Collect and print statistics for each client
    client_details = []
    
    for client_id, (train_dict, test_dict) in enumerate(zip(train_data, test_data)):
        y_train = np.array(train_dict['y'])
        y_test = np.array(test_dict['y'])
        
        # Calculate label distributions
        train_labels, train_counts = np.unique(y_train, return_counts=True)
        test_labels, test_counts = np.unique(y_test, return_counts=True)
        
        # Store client details
        client_details.append({
            'client_id': client_id,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'train_label_distribution': dict(zip(train_labels.tolist(), train_counts.tolist())),
            'test_label_distribution': dict(zip(test_labels.tolist(), test_counts.tolist()))
        })
        
        # Print dataset details for each client
        print(f"Client {client_id}\t Size of training data: {len(y_train)}\t Size of testing data: {len(y_test)}")
        print(f"\t\t Training Labels: {np.unique(y_train)}")
        print(f"\t\t Testing Labels: {np.unique(y_test)}")
        print("-" * 50)
    
    # Save the dataset to disk
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        client_details, niid, balance, partition, alpha, few_shot, n_shot, pfl)


if __name__ == "__main__":
    # Check if the minimum number of arguments is provided
    if len(sys.argv) < 7:
        print("Usage: script.py num_clients niid balance partition alpha few_shot [n_shot]")
        sys.exit(1)

    # Parse arguments
    try:
        num_clients = int(sys.argv[1])
    except ValueError:
        print("Invalid input for num_clients. Please provide an integer value.")
        sys.exit(1)

    niid = sys.argv[2].lower() == "noniid"
    balance = sys.argv[3].lower() == "balance"
    partition = sys.argv[4]
            
    # Alpha is required only for non-IID data with "dir" partition
    alpha = None
    if niid and partition == "dir":
        if len(sys.argv) < 6 or sys.argv[5] == "-":
            print("Alpha parameter is required for non-IID 'dir' partitioned data.")
            sys.exit(1)
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value.")
            sys.exit(1)
    elif len(sys.argv) >= 6 and sys.argv[5] != "-":
        # Optional alpha for other cases
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value or '-' for default.")
            sys.exit(1)

    few_shot = sys.argv[6].lower() in ["true", "fs"]

    n_shot = None
    if few_shot:
        if len(sys.argv) < 8:
            print("n_shot parameter is required for few_shot mode.")
            sys.exit(1)
        try:
            n_shot = int(sys.argv[7])
        except ValueError:
            print("Invalid input for n_shot. Please provide an integer value.")
            sys.exit(1)
            
    pfl = sys.argv[8].lower() == "pfl"
    
    # Print all parsed arguments
    print(f"Running script with the following parameters:")
    print(f"num_clients: {num_clients}")
    print(f"niid: {niid}")
    print(f"balance: {balance}")
    print(f"partition: {partition}")
    print(f"alpha: {alpha}")
    print(f"few_shot: {few_shot}")
    print(f"n_shot: {n_shot}")
    print(f"pfl: {pfl}")

    generate_aircraft(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot, pfl)