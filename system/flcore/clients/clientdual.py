import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
# from transformers import AdamW
from flcore.optimizers.sparse_optimizer import SparseAdamW

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import return_zeroshot_weight, accuracy, read_client_data_clip
from torch.utils.data import Subset

from pathlib import Path

from flcore.trainmodel.clip_model_dual import *

class ClientDualLORA(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        # dual‐LoRA CLIP
        cache_dir = Path(args.home_dir) / "models"
        self.clip_model_object = CLIPModelWithDualLoRA(
            checkpoint=args.model_checkpoint,
            home_dir=args.home_dir,
            lora_params_global=args.lora_params_global,
            lora_params_local =args.lora_params_local,
        )
        self.clip_model = self.clip_model_object.model_combined

        # collect parameter dicts
        self.global_adapters = self.clip_model_object.lora_layers_global
        self.local_adapters  = self.clip_model_object.lora_layers_local
        self.gating_params   = self.clip_model_object.lora_gating

        # optimizer covers global + local + gating
        params = list(self.global_adapters.values()) \
               + list(self.local_adapters.values())  \
               + list(self.gating_params.values())

        self.optimizer = torch.optim.AdamW(
            params=params,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=args.learning_rate_decay_gamma
        )

        self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()

    # def set_parameters(self, global_dict):
    #     # load server’s global adapter
    #     for k, p in self.global_adapters.items():
    #         p.data.copy_(global_dict[k].data)

    def get_parameters(self):
        # send back only global adapter
        return {k: v.clone().detach() for k, v in self.global_adapters.items()}

    def train(self):
        loader = self.load_train_data()
        self.clip_model.to(self.device).train()
        t0 = time.time()

        for epoch in range(self.local_epochs):
            for images, _, texts in tqdm(loader, desc=f"Client {self.id} Epoch {epoch+1}"):
                images = images.to(self.device)
                input_ids      = texts['input_ids'].squeeze(1).to(self.device)
                attention_mask = texts['attention_mask'].squeeze(1).to(self.device)

                # forward through combined CLIP+dual‐LoRA
                img_feats = self.clip_model.get_image_features(images).float()
                txt_feats = self.clip_model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                ).float()
                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)

                logits_i2t = self.logit_scale * img_feats @ txt_feats.t()
                logits_t2i = logits_i2t.t()
                gt = torch.arange(len(images), device=self.device)

                loss = (self.loss(logits_i2t, gt) + self.loss(logits_t2i, gt)) / 2

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
                self.optimizer.step()

        # stepping scheduler & logging
        if self.learning_rate_decay:
            self.lr_scheduler.step()
        elapsed = time.time() - t0
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost']  += elapsed
        print(f"Client {self.id} train time {elapsed/60:.2f}m, mem {torch.cuda.max_memory_reserved()/1e9:.2f}GB")

        # 1) after all epochs, gather gating scalars from every LoRALayer
        gs = []
        for module in self.clip_model.modules():
            if isinstance(module, LinearWithDualLoRA) and module.last_gating is not None:
                # take the mean over batch (and tokens, if any)
                gs.append(module.last_gating.mean().item())
        # 2) average them to a single float
        avg_g = sum(gs) / len(gs) if len(gs) > 0 else 0.0

        # 3) print or store it
        print(f"[Client {self.id}] avg gating = {avg_g:.4f}")

        # 4) return it to the server
        return avg_g

    def test_metrics(self):
        testloader = self.load_test_data()
        self.clip_model.to(self.device)
        self.clip_model.eval()
                
        with torch.no_grad():
            top1_1, test_num = 0., 0

            for i, (images, target, texts) in enumerate(testloader):
                images = images.to(self.device)
                target = target.to(self.device)
                texts = texts.to(self.device)

                # predict
                image_features = self.clip_model.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # measure accuracy of 1 template
                zeroshot_weights_1 = return_zeroshot_weight(self.dataset, self.clip_model, self.processor, self.class_names, self.device)
                logits = self.logit_scale * image_features @ zeroshot_weights_1
                acc1 = accuracy(logits, target, topk=(1,))
                top1_1 += acc1[0]

                test_num += images.size(0)
                
        self.clip_model.to("cpu")
        
        # top1_1 = (top1_1 / test_num) 
        print(f"Number of testing samples: {test_num}")
        # print(f"Top-1 accuracy: {top1_1:.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        return top1_1, test_num
                
    def set_parameters(self, dictionary):
        self.clip_model_object.set_lora_dict(dictionary)

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    