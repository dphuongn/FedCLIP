import time
import copy
import random
from flcore.clients.clientmeta import clientMeta
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch

from flcore.trainmodel.clip_model_simple import *


# --- MetaLoRA Server ---
class FMeta(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.clip_model_object = CLIPModelWithLoRA(
            args.model_checkpoint, args.home_dir, args.lora_params
        )
        self.global_model      = copy.deepcopy(self.clip_model_object.lora_layers)
        layers = self.clip_model_object.lora_layers
        self.meta_params       = [p for n,p in layers.items() if n.endswith('.W_a') or n.endswith('.W_b')]
        self.gate_params       = [p for n,p in layers.items() if n.endswith('.gate')]
        self.server_lr         = args.server_lr
        self.client_updates    = []

        self.set_clients(clientMeta)

        # attach server reference onto each client
        for c in self.clients:
            c.server = self

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if not self.pfl:
                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate()
                if i < self.global_rounds:
                    for client in self.selected_clients:
                        client.train()
                    self.receive_models()
                    self.aggregate_meta()
            else:
                print(f"\n-------------Round number: {i}-------------")
                for client in self.selected_clients:
                    client.train()
                print("\n-------------Evaluate personalized models-------------")
                self.evaluate()
                self.receive_models()
                self.aggregate_meta()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def send_models(self):
        assert len(self.clients) > 0

        for c in self.clients:
            t0 = time.time()
            c.set_parameters(self.global_model)
            c.send_time_cost['num_rounds'] += 1
            c.send_time_cost['total_cost']  += 2 * (time.time() - t0)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        # active = random.sample(
        #     self.selected_clients,
        #     int((1 - self.client_drop_rate) * self.current_num_join_clients)
        # )
        # self.uploaded_weights, self.uploaded_models = [], []
        # total_samples = 0
        # for c in active:
        #     total_samples += c.train_samples
        #     self.uploaded_weights.append(c.train_samples)
        #     self.uploaded_models.append(c.clip_model_object.lora_layers)
        # if not self.no_normalize_weights:
        #     self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        # else:
        #     self.uploaded_weights = [1.0 for _ in self.uploaded_weights]

        active = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )
        total = sum(c.train_samples for c in active)
        self.uploaded_weights = [c.train_samples for c in active]
        self.uploaded_models  = [c.clip_model_object.lora_layers for c in active]

        if not self.no_normalize_weights:
            self.uploaded_weights = [w/total for w in self.uploaded_weights]
        else:
            self.uploaded_weights = [1.0]*len(active)

    def aggregate_parameters_lora(self):
        assert len(self.uploaded_models) > 0
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for p in self.global_model.values(): p.data.zero_()
        for w, cm in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_lora(w, cm)

    # def add_parameters_lora(self, w, client_model):
    #     for name, gp in self.global_model.items():
    #         gp.data += client_model[name].data * w

    def add_parameters_lora(self, w, cm):
        # for backward compatibility; not used in meta
        for k,g in self.global_model.items():
            g.data += cm[k].data * w

    def receive_meta_updates(self, update_dict):
        # append into the same list you’ll use in aggregate_meta()
        self.client_updates.append(update_dict)

    def aggregate_meta(self):
        if not self.client_updates:
            return
        n = len(self.client_updates)
        # 1) average each client’s update dict (these tensors might be on CUDA)
        avg = {
            k: sum(d[k] for d in self.client_updates) / n
            for k in self.client_updates[0]
        }
        # 2) take a partial step toward that avg, respecting old device
        new_global = {}
        for k, old_p in self.global_model.items():
            # cast the avg[k] onto old_p’s device
            update = avg[k].to(old_p.device)
            new_global[k] = old_p + self.server_lr * (update - old_p)
        # 3) swap in and write back to your CLIPModelWithLoRA
        self.global_model = new_global
        self.clip_model_object.set_lora_dict(new_global)
        self.client_updates.clear()


    # def aggregate_meta(self):
    #     if not self.client_updates:
    #         return
    #     n = len(self.client_updates)
    #     # average client dicts
    #     avg = {
    #         k: sum(d[k] for d in self.client_updates)/n
    #         for k in self.client_updates[0]
    #     }
    #     # with server_lr step
    #     new_global = {}
    #     for k, old_p in self.global_model.items():
    #         new_global[k] = old_p + self.server_lr*(avg[k] - old_p)
    #     self.global_model = new_global
    #     # write back into the CLIPModelWithLoRA
    #     self.clip_model_object.set_lora_dict(new_global)
    #     self.client_updates.clear()

    # def aggregate_meta(self):
    #     if not self.client_updates:
    #         return
    #     n = len(self.client_updates)
    #     # average each parameter across dicts
    #     avg = {
    #         k: sum(d[k] for d in self.client_updates) / n
    #         for k in self.client_updates[0]
    #     }
    #     # load back into your global LoRA dict
    #     self.clip_model_object.set_lora_dict(avg)
    #     self.client_updates.clear()

# # --- Server with Meta-Gradient Aggregation ---
# class FMeta(Server):
#     def __init__(self, args, times):
#         super().__init__(args, times)

#         self.lora_params = args.lora_params
#         self.clip_model_object = CLIPModelWithLoRA(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params=self.lora_params)

#         self.global_model = copy.deepcopy(self.clip_model_object.lora_layers)    # dict of params

#         # extract global meta params from lora_layers_global dict
#         layers = self.clip_model_object.lora_layers  # use the unified lora_layers dict from CLIPModelWithLoRA
#         self.meta_params = [p for name, p in layers.items() if name.endswith('.W_a') or name.endswith('.W_b')]
#         self.gate_params = [p for name, p in layers.items() if name.endswith('.gate')]
#         # self.server_meta_optimizer = AdamW(self.meta_params + self.gate_params, lr=args.meta_server_lr)
#         self.client_updates = []

#         self.set_clients(clientMeta)

#         print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
#         print("Finished creating server and clients.")

#         self.Budget = []

#     def receive_meta_updates(self, update_dict):
#         # Collect clients' meta-gradient updates
#         self.client_meta_updates.append(update_dict)

#     def aggregate_meta(self):
#         # average client updates and apply
#         n = len(self.client_updates)
#         avg = {k: sum(d[k] for d in self.client_updates) / n for k in self.client_updates[0]}
#         # load averaged into model
#         self.clip_model_object.set_lora_dict_global(avg)
#         self.client_updates.clear()

#     def train(self):
#         for i in range(self.global_rounds + 1):
#             s_t = time.time()
#             self.selected_clients = self.select_clients()
#             self.send_models()

#             if not self.pfl: # tfl
#                 #=========== traditional FL ===========
#                 if i%self.eval_gap == 0:
#                     print(f"\n-------------Round number: {i}-------------")
#                     print("\nEvaluate global model")
#                     self.evaluate()

#                 if i < self.global_rounds: # skip training for the last round to save time

#                     for client in self.selected_clients:
#                         client.train()

#                     self.receive_models()

#                     # self.aggregate_parameters_lora()
#                     self.aggregate_meta()
#             else:
#                 #=========== personalized FL ===========
#                 print(f"\n-------------Round number: {i}-------------")
#                 for client in self.selected_clients:
#                     client.train()

#                 print("\n-------------Evaluate personalized models-------------")
#                 self.evaluate()

#                 self.receive_models()

#                 # self.aggregate_parameters_lora()
#                 self.aggregate_meta()

#             self.Budget.append(time.time() - s_t)
#             print('-'*25, 'time cost', '-'*25, self.Budget[-1])

#         print("\nBest accuracy.")
#         print(max(self.rs_test_acc))
#         print("\nAverage time cost per round.")
#         print(sum(self.Budget[1:])/len(self.Budget[1:]))

#         self.save_results()


#     def send_models(self):
#         # Instead of sending the whole model, only send LoRA layers
#         assert (len(self.clients) > 0)

#         for client in self.clients:
#             start_time = time.time()
            
#             global_lora_params = self.global_model
            
#             client.set_parameters(global_lora_params)

#             client.send_time_cost['num_rounds'] += 1
#             client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
#     def receive_models(self):
#         # Receive only the LoRA layers from each client
#         assert (len(self.selected_clients) > 0)

#         active_clients = random.sample(
#             self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

#         self.uploaded_weights = []
#         self.uploaded_models = []
#         total_samples = 0
#         for client in active_clients:
#             client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
#                     client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            
#             total_samples += client.train_samples
#             self.uploaded_weights.append(client.train_samples)
#             self.uploaded_models.append(client.lora_layers)
            
#         if not self.no_normalize_weights: # normalize weights as usual
#             self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
#         else: # no need for normalize weights
#             self.uploaded_weights = [1.0 for _ in self.uploaded_weights]
            
#     def aggregate_parameters_lora(self):
#         assert (len(self.uploaded_models) > 0)

#         # Initialize global model
#         self.global_model = copy.deepcopy(self.uploaded_models[0])
        
#         # Zero out the global model parameters
#         for param in self.global_model.values():
#             param.data.zero_()
            
#         # Aggregate parameters
#         for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
#             self.add_parameters_lora(w, client_model)
            
#     def add_parameters_lora(self, w, client_model):
#         # Perform weighted aggregation of parameters
#         for name, global_param in self.global_model.items():
#             global_param.data += client_model[name].data * w
    
    
            
    
    
    
    
    