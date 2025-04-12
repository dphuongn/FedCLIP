import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.utils.data import Subset
from utils.data_utils import read_client_data_clip, accuracy
import wandb  # Import wandb


class Client(object):
    """
    Base class for clients in federated learning.
    """

    # def __init__(self, args, wandb_config, id, **kwargs):
    def __init__(self, args, id, **kwargs):
        # self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.global_rounds = args.global_rounds

        self.num_classes = args.num_classes
        
        self.train_samples = 0
        self.test_samples = 0
        
        self.batch_size_train = args.batch_size_train
        self.batch_size_test = args.batch_size_test

        # Use wandb_config for hyperparameters
        # self.learning_rate = wandb_config.local_learning_rate
        self.learning_rate = args.local_learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.eps = args.eps
        # self.weight_decay = wandb_config.weight_decay
        self.weight_decay = args.weight_decay
        # self.learning_rate_decay = wandb_config.learning_rate_decay
        self.learning_rate_decay = args.learning_rate_decay
        self.local_epochs = args.local_epochs
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        
        self.class_names = args.class_names
        self.train_data_fraction = args.train_data_fraction
        self.test_data_fraction = args.test_data_fraction

        self.processor = args.processor

        self.loss = nn.CrossEntropyLoss()
        
    def load_train_data(self, batch_size=None, train_data_fraction_blockprune=None):
        if batch_size == None:
            batch_size = self.batch_size_train
        train_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True)
        
        if train_data_fraction_blockprune is not None:
            self.train_samples = int(len(train_data) * self.train_data_fraction_blockprune)
        else:
            self.train_samples = int(len(train_data) * self.train_data_fraction)
        
        train_indices = np.random.choice(len(train_data), self.train_samples, replace=False)
        train_subset = Subset(train_data, train_indices)
        
        return DataLoader(train_subset, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None, id=None, test_data_fraction_blockprune=None):
        if batch_size == None:
            batch_size = self.batch_size_test
        if id is not None:
            test_data = read_client_data_clip(self.dataset, id, self.processor, self.class_names, self.device, is_train=False)
        else:
            test_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=False)

        if test_data_fraction_blockprune is not None:
            self.test_samples = int(len(test_data) * self.test_data_fraction_blockprune)
        else:
            self.test_samples = int(len(test_data) * self.test_data_fraction)
        
        test_indices = np.random.choice(len(test_data), self.test_samples, replace=False)
        test_subset = Subset(test_data, test_indices)
        
        return DataLoader(test_subset, batch_size, drop_last=False, shuffle=False)    