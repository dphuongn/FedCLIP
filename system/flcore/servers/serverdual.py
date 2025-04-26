import time
import copy
import numpy as np
from flcore.clients.clientdual import ClientDualLORA
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch

from torch.utils.data import DataLoader, Subset
from utils.data_utils import read_client_data_clip

from flcore.trainmodel.clip_model_dual import *


class FLoraDual(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # two sets of LoRA params
        self.lora_params_global = args.lora_params_global
        self.lora_params_local  = args.lora_params_local

        self.learning_rate = args.local_learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.eps = args.eps
        self.weight_decay = args.weight_decay

        # build a combined CLIP with dual LoRA
        cache_dir = Path(args.home_dir) / "models"
        self.clip_model_object = CLIPModelWithDualLoRA(
            checkpoint=args.model_checkpoint,
            home_dir=args.home_dir,
            lora_params_global=self.lora_params_global,
            lora_params_local=self.lora_params_local,
        )

        # initialize the server’s global adapter dict
        self.global_model = copy.deepcopy(self.clip_model_object.lora_layers_global)

        # set up clients
        self.set_clients(ClientDualLORA)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

        self.logit_scale = self.clip_model_object.model_combined.state_dict()['logit_scale'].exp()
        # load your small reference set once
        # self.ref_loader = DataLoader(self.ref_dataset, batch_size=self.batch_size_train, shuffle=True)

    def train(self):
        for rnd in range(self.global_rounds + 1):
            t0 = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            #  (T) or (P)-FL logic unchanged, but they all call client.train()
            if not self.pfl:
                if rnd % self.eval_gap == 0:
                    print(f"\n-------------Round number: {rnd}-------------")
                    print("\nEvaluate global model")
                    self.evaluate()
                if rnd < self.global_rounds:
                    for c in self.selected_clients: 
                        c.train()
                    self.receive_models()
                    # self.aggregate_parameters_lora()
                    self.aggregate_via_distillation()

            else:
                # print(f"\n----- Round {rnd} (Personalized) -----")
                print(f"\n-------------Round number: {rnd}-------------")
                for c in self.selected_clients: 
                    # c.train()
                    g_val = c.train()
                    # this will print client‐side already, 
                    # but you can also aggregate or print server‐side:
                    print(f"Round {rnd} → Client {c.id} gating: {g_val:.4f}")
                print("\n-------------Evaluate personalized models-------------")
                self.evaluate()
                self.receive_models()
                # self.aggregate_parameters_lora()
                self.aggregate_via_distillation()

            self.Budget.append(time.time() - t0)
            print(f"{'-'*10} round time: {self.Budget[-1]:.2f}s {'-'*10}")

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("Avg time per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def send_models(self):
        # push only the server’s global LoRA dict
        for c in self.clients:
            t0 = time.time()
            c.set_parameters(self.global_model)
            c.send_time_cost['num_rounds'] += 1
            c.send_time_cost['total_cost']  += 2 * (time.time() - t0)

    def receive_models(self):

        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in self.selected_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            
            total_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.global_adapters)
            
        if not self.no_normalize_weights: # normalize weights as usual
            self.uploaded_weights = [w / total_samples for w in self.uploaded_weights]
        else: # no need for normalize weights
            self.uploaded_weights = [1.0 for _ in self.uploaded_weights]

    def load_ref_data(self, batch_size=None, ref_data_fraction=None):
        if batch_size == None:
            batch_size = self.batch_size_ref
        ref_data = read_client_data_clip('cifar10', 1, self.processor, self.class_names, self.device, is_train=True)
        
        if ref_data_fraction is not None:
            self.ref_samples = int(len(ref_data) * self.ref_data_fraction)
        else:
            self.ref_samples = int(len(ref_data))
        
        ref_indices = np.random.choice(len(ref_data), self.ref_samples, replace=False)
        ref_subset = Subset(ref_data, ref_indices)
        
        return DataLoader(ref_subset, batch_size, drop_last=False, shuffle=False)

    # def aggregate_via_distillation(self, distill_epochs=1, lr=1e-3):
    def aggregate_via_distillation(self, distill_epochs=1, lr=self.learning_rate):

        self.ref_loader = self.load_ref_data()

        # 1) collect client global‐only dicts
        # client_adapters = [c.get_parameters() for c in self.selected_clients]

        # 1) use the already-received adapters + normalized weights
        client_adapters = self.uploaded_models       # list of dicts
        client_weights  = self.uploaded_weights      # list of floats, sum=1

        # 2) freeze backbone + local + gating
        self.clip_model_object.freeze_backbone_and_local()
        self.clip_model_object.set_global_only(True)   # only global used in forward

        params = list(self.global_adapters.values())
        # opt = torch.optim.AdamW(
        #     self.clip_model_object.lora_layers_global.values(), lr=lr
        # )
        opt = torch.optim.AdamW(
            params=params,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        # 3) Inner distillation loop
        for _ in range(distill_epochs):
            for images, _, texts in self.ref_loader:
                
                images = images.to(self.device)
                input_ids = texts['input_ids'].squeeze(1).to(self.device)
                attn_mask = texts['attention_mask'].squeeze(1).to(self.device)

                # 3a) Build teacher probabilities
                # 3) teacher logits: average of each client’s fine-tuned global‐LoRA on ref batch
                
                teacher_probs = []
                with torch.no_grad():
                    # for adapter_dict in client_adapters:
                    for w, adapter_dict in zip(client_weights, client_adapters):
                        # copy wrapper, load that client’s global‐only weights
                        teacher = copy.deepcopy(self.clip_model_object)
                        teacher.set_global_adapter(adapter_dict)
                        # teacher is already in global‐only mode

                        teacher.to(self.device)

                        # forward through teacher
                        i_feats = teacher.model_combined.get_image_features(images).float()
                        t_feats = teacher.model_combined.get_text_features(
                            input_ids=input_ids, attention_mask=attn_mask
                        ).float()
                        i_feats = F.normalize(i_feats, dim=1)
                        t_feats = F.normalize(t_feats, dim=1)
                        logits = self.logit_scale * (i_feats @ t_feats.t())

                        # teacher_probs.append(F.softmax(logits, dim=-1))
                        # weight each client’s distribution
                        teacher_probs.append(w * F.softmax(logits, dim=-1))

                    # average over clients → [B, C]
                    # T = torch.stack(teacher_probs, dim=0).mean(0)
                    # weighted sum → [B, C]
                    T = torch.stack(teacher_probs, dim=0).sum(0)

                self.clip_model_object.model_combined.to(self.device).train()

                # 3b) student logits (uses same student global‐only wrapper)
                s_i = self.clip_model_object.model_combined.get_image_features(images).float()
                s_t = self.clip_model_object.model_combined.get_text_features(
                    input_ids=input_ids, attention_mask=attn_mask
                ).float()
                s_i = F.normalize(s_i, dim=1)
                s_t = F.normalize(s_t, dim=1)
                student_logits = self.logit_scale * (s_i @ s_t.t())
                S = F.log_softmax(student_logits, dim=-1)

                # 3c) KL loss + step
                loss = kl_loss(S, T)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # 4) restore local LoRA & gating for next FL round
            self.clip_model_object.set_global_only(False)
            self.clip_model_object.unfreeze_local_and_gate()

            teacher.to("cpu")
            self.clip_model_object.model_combined.to("cpu")

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