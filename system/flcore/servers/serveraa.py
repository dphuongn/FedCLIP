import time
import copy
import random
from flcore.clients.clientaa import clientAA
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch

from flcore.trainmodel.clip_model import *


class FedAa(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.aa_params = args.aa_params
        
        self.clip_model_object = CLIPModelWithAttentionAdapter(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, aa_params=self.aa_params)
        
        self.global_model = copy.deepcopy(self.clip_model_object.aa)    # aa model
        
        self.set_clients(clientAA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []


    def train(self):
        
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f'self.current_num_join_clients: {self.current_num_join_clients}')
            self.send_models()

            if not self.pfl: # tfl
                #=========== traditional FL ===========
                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate()

                if i < self.global_rounds: # skip training for the last round to save time

                    for client in self.selected_clients:
                        client.train()

                    self.receive_models()

                    self.aggregate_parameters_aa()
            else:
                #=========== personalized FL ===========
                print(f"\n-------------Round number: {i}-------------")
                for client in self.selected_clients:
                    client.train()

                print("\n-------------Evaluate personalized models-------------")
                self.evaluate()

                self.receive_models()

                self.aggregate_parameters_aa()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAA)
            if self.finetune_new_clients:
                print(f"\n-------------Fine tuning round-------------")
                self.fine_tuning_new_clients()
            print("\nEvaluate new clients")
            self.evaluate_new_clients()
    
    def send_models(self):
        # Instead of sending the whole model, only send the Attention Adapter 
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            global_aa_params = self.global_model
            
            client.set_parameters(global_aa_params)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def receive_models(self):
        # Receive only the Attention Adapter from each client
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        print(f'active_clients: {len(active_clients)}')

        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            
            
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.aa)
            
        if not self.no_normalize_weights: # normalize weights as usual
            self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        else: # no need for normalize weights
            self.uploaded_weights = [1.0 for _ in self.uploaded_weights]
            
    def aggregate_parameters_aa(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_aa(w, client_model)
            
    def add_parameters_aa(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
            
            
    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)                
            client.finetune()
            
    
    