import time
import copy
import random
from flcore.clients.clientloraselecthead import clientLORASELECTHEAD
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch

from flcore.trainmodel.clip_model_drop import *


class FLoraSelectHead(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.lora_params = args.lora_params
        
        self.clip_model_object = CLIPModelWithLoRA(
            model_checkpoint=args.model_checkpoint, 
            home_dir=args.home_dir, 
            lora_params=self.lora_params
        )
        
        # Initialize global LoRA model as an empty dictionary
        self.global_model = {}
        
        self.set_clients(clientLORASELECTHEAD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def initialize_client_layers(self):
        """Calculate and remove least important layers for all clients before the first iteration."""
        print("\nInitializing clients with layer importance evaluation...")
        for client in self.clients:
            client.send_projection_layers()


    def train(self):

        # Perform client initialization before training begins
        self.initialize_client_layers()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models()

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
                
            # move send_models at the end to handle heterogenous LoRA layers    
            self.send_models()
                
                
            """
            Check and display the number of LoRA layers for each client.
            """
            for client in self.clients:
                num_layers = client.clip_model_object.get_number_of_lora_layers()
                print(f"Client {client.id} has {num_layers} LoRA layers.")
            
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientLORA)
            
            print(f"\n-------------Evaluate zero shot new client-------------")
            self.evaluate_new_clients()
            
            if self.finetune_new_clients:
                print(f"\n-------------Fine tuning round-------------")
                self.fine_tuning_new_clients()
            print("\nEvaluate new clients")
            self.evaluate_new_clients()

    def send_models(self):
        """Send the global LoRA parameters to clients."""
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            # Filter global_model based on client's initial remained layers
            filtered_global_model = {
                key: param
                for key, param in self.global_model.items()
                if key in client.lora_layers_select
            }

            # print(f'filtered_global_model for client {client.id}:')
            # for key in filtered_global_model.keys():
            #     print(f"{key}")
            
            print(f'set paramters select for client {client.id}:')
            client.set_parameters_select(filtered_global_model)  # Send filtered LoRA parameters
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
            # client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #         client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.lora_layers_select)

        # print(f'self.uploaded_models: {self.uploaded_models}')

        # Inspect the structure of LoRALayer in uploaded_models
        # for i, client_model in enumerate(self.uploaded_models):
        #     print(f"\nClient {i+1} uploaded model:")
        #     for key, layer in client_model.items():
        #         print(f"  Layer: {key}")
        #         print(f"  Type: {type(layer)}")
        #         if isinstance(layer, LoRALayer):
        #             print(f"  Attributes of LoRALayer {key}:")
        #             for attr in dir(layer):
        #                 if not attr.startswith("_"):  # Exclude private/internal attributes
        #                     print(f"    {attr}: {getattr(layer, attr, 'N/A')}")
        #         else:
        #             print(f"  Non-LoRALayer value: {layer}")
            
        if not self.no_normalize_weights: # normalize weights as usual
            self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        else: # no need for normalize weights
            self.uploaded_weights = [1.0 for _ in self.uploaded_weights]
            
    def aggregate_parameters_lora(self):
        assert (len(self.uploaded_models) > 0)

        # # Initialize global model
        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        
        # Initialize global model with keys from all clients
        self.global_model = {}
        
        # Collect all unique layers from clients
        all_layer_keys = set()
        for client_model in self.uploaded_models:
            all_layer_keys.update(client_model.keys())
            
        # Initialize global model for each layer
        # for key in all_layer_keys:
        #     self.global_model[key] = torch.zeros_like(next(iter(self.uploaded_models[0].values())))

        # Initialize global model for each layer
        for key in all_layer_keys:
            # Find the first matching layer from clients
            for client_model in self.uploaded_models:
                if key in client_model:
                    self.global_model[key] = torch.zeros_like(client_model[key])
                    break

                    # Initialize entries for W_a and W_b
                    # self.global_model[f"{key}.W_a"] = torch.zeros_like(client_model[key].W_a)
                    # self.global_model[f"{key}.W_b"] = torch.zeros_like(client_model[key].W_b)
                    # break
            
        # Aggregate parameters
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_lora(w, client_model)

    def add_parameters_lora(self, w, client_model):
        # Perform weighted aggregation of parameters
        for name, global_param in self.global_model.items():
            if name in client_model:  # Check if the layer exists in the client model
                client_param = client_model[name]

                # Ensure both tensors are on the same device
                if global_param.device != client_param.device:
                    client_param = client_param.to(global_param.device)
                    
                # Ensure size compatibility
                if global_param.size() == client_param.size():
                    global_param.data += client_param.data * w
                else:
                    print(f"Dimension mismatch for layer {name}: global {global_param.size()}, client {client_param.size()}")
                    raise ValueError(f"Mismatch in dimensions for layer {name}")

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)                
            client.finetune()
    
    
            
    
    
    
    
    