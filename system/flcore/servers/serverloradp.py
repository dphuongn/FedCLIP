import time
import copy
import random
from flcore.clients.clientloradp import clientLORADP
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch

from flcore.trainmodel.clip_model import *


class FLoradp(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.lora_params = args.lora_params
        
        self.clip_model_object = CLIPModelWithLoRA(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params=self.lora_params)
        
        self.global_model = copy.deepcopy(self.clip_model_object.lora_layers)    # dict of params
        
        self.set_clients(clientLORADP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.saved_privacy_engine_state = None  # Variable to store PrivacyEngine state

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
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

                    self.aggregate_parameters_lora()
            else:
                #=========== personalized FL ===========
                print(f"\n-------------Round number: {i}-------------")
                for client in self.selected_clients:
                    client.train()

                print("\n-------------Evaluate personalized models-------------")
                self.evaluate()

                self.receive_models()

                self.aggregate_parameters_lora()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientLORADP)
            if self.finetune_new_clients:
                print(f"\n-------------Fine tuning round-------------")
                self.fine_tuning_new_clients()
            print("\nEvaluate new clients")
            self.evaluate_new_clients()
        
    def send_models(self):
        # Instead of sending the whole model, only send LoRA layers
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            global_lora_params = self.global_model
            
            client.set_parameters(global_lora_params)

            # Restore saved PrivacyEngine state to the client if available
            if self.saved_privacy_engine_state:
                client.set_privacy_engine_state(self.saved_privacy_engine_state)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def receive_models(self):
        # Receive only the LoRA layers from each client
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.lora_layers)

            # Save the PrivacyEngine state from the client

            self.saved_privacy_engine_state = client.get_privacy_engine_state()
            # if hasattr(client.clip_model, "privacy_engine") and client.clip_model.privacy_engine:
            #     self.saved_privacy_engine_state = client.clip_model.privacy_engine.get_state() # Save the PrivacyEngine state
            
        if not self.no_normalize_weights: # normalize weights as usual
            self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        else: # no need for normalize weights
            self.uploaded_weights = [1.0 for _ in self.uploaded_weights]
            
    def aggregate_parameters_lora(self):
        assert (len(self.uploaded_models) > 0)

        # Initialize global model
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        
        # Zero out the global model parameters
        for param in self.global_model.values():
            param.data.zero_()
            
        # Aggregate parameters
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_lora(w, client_model)
            
    def add_parameters_lora(self, w, client_model):
        # Perform weighted aggregation of parameters
        for name, global_param in self.global_model.items():
            global_param.data += client_model[name].data * w

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)                
            client.finetune()
    
    
            
    
    
    
    
    