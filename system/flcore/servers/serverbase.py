import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import statistics
import wandb  # Import wandb


class Server(object):
    # def __init__(self, args, times, wandb_config):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        # self.wandb_config = wandb_config
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        
        self.processor = args.processor
        self.class_names = args.class_names
        self.no_normalize_weights = args.no_normalize_weights
        self.pfl = args.personalized_fl

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        
        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.finetune_new_clients = args.finetune_new_clients
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        
        
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            client = clientObj(self.args,
                               id=i
                              )
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

            
    def test_metrics_tfl(self):
        num_samples = []
        tot_correct = []
            
        c = self.clients[0]    
        
        ct, ns = c.test_metrics()
        tot_correct.append(ct*1.0)
        num_samples.append(ns)
            
        print(f'Client {c.id}: Test Accuracy: {ct*1.0/ns}')
        
        ids = [c.id]
        
        # print(f'ids: {ids}')

        return ids, num_samples, tot_correct

    def test_metrics_pfl(self):
        num_samples = []
        tot_correct = []
        
        for c in self.clients:
            ct, ns = c.test_metrics()
            print(f'Client {c.id}: Test Accuracy: {ct*1.0/ns}')
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        
        # print(f'ids: {ids}')

        return ids, num_samples, tot_correct
    
    def train_metrics_tfl(self):
        
        num_samples = []
        losses = []
        
        c = self.clients[0]    
        
        cl, ns = c.train_metrics()
        num_samples.append(ns)
        losses.append(cl*1.0)
        
        print(f'Client {c.id}: Train loss: {cl*1.0/ns}')

        ids = [c.id]

        return ids, num_samples, losses

    def train_metrics_pfl(self):
        
        num_samples = []
        losses = []
        
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl*1.0/ns}')
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    
    
    def evaluate(self, acc=None, loss=None):
        
        if not self.pfl: # tfl
            stats_test = self.test_metrics_tfl()
            # stats_train = self.train_metrics_tfl()      # comment this for without train loss (faster)
            
            # print(f'Number of testing samples: {stats_test[1]}')
            # print(f'Top-1 testing accuracy: {stats_test[2]}')
            
            # test_acc = sum(stats_test[2])*1.0 / len(stats_test[2])
            test_acc = sum(stats_test[2])*1.0 / sum(stats_test[1])
            # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])      # comment this for without train loss (faster)
            accs = [a / n for a, n in zip(stats_test[2], stats_test[1])] 
            
            # accs = 0
            
        else: # pfl
            stats_test = self.test_metrics_pfl()
            # stats_train = self.train_metrics_pfl()         # comment this for without train loss (faster)
        
            # print(f'Number of testing samples: {stats_test[1]}')
            # print(f'Top-1 testing accuracy: {stats_test[2]}')

            test_acc = sum(stats_test[2])*1.0 / sum(stats_test[1])
            # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])    # comment this for without train loss (faster)
            accs = [a / n for a, n in zip(stats_test[2], stats_test[1])]

            # accs = statistics.stdev(stats[2])
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
            
        # comment this for without train loss (faster)
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)
        
        # print("Averaged Train Loss: {:.4f}".format(train_loss))     # comment this for without train loss (faster)
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        
        # wandb.log({"Averaged Train Loss": train_loss, "Averaged Test Accuracy": test_acc, "Std Test Accurancy": np.std(accs)})
    
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
                
    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            client = clientObj(self.args, 
                            id=i
                              )
            self.new_clients.append(client)
            
    def evaluate_new_clients(self, acc=None, loss=None):
    
        stats_test = self.test_metrics_new_clients()

        test_acc = sum(stats_test[2])*1.0 / sum(stats_test[1])
        accs = [a / n for a, n in zip(stats_test[2], stats_test[1])]
        
        print("New Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("New Std Test Accurancy: {:.4f}".format(np.std(accs)))
                    
    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns = c.test_metrics()
            print(f'New client {c.id}: Test Accuracy: {ct*1.0/ns}')
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct
    