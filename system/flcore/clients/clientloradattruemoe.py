import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
# from transformers import AdamW

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import return_zeroshot_weight, accuracy
from torch.utils.data import Subset

from pathlib import Path

from flcore.trainmodel.clip_model_moe import *


class clientLORADATTRUEMOE(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        
        self.lora_params = args.lora_params

        self.moe_params = args.moe_params
        
        self.momentum_global = args.momentum_global
        self.momentum_local = args.momentum_local
        
        self.mu_global = torch.nn.Parameter(torch.tensor(args.mu_global, dtype=torch.float32))
        self.mu_learning_rate_global = args.mu_learning_rate_global
        
        self.mu_local = torch.nn.Parameter(torch.tensor(args.mu_local, dtype=torch.float32))
        self.mu_learning_rate_local = args.mu_learning_rate_local
        
        self.clip_model_object = CLIPModelWithLoRACombinedMoE(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params_global=self.lora_params, lora_params_local=self.lora_params, moe_params=self.moe_params, momentum_global=self.momentum_global, momentum_local=self.momentum_local)
        
        self.clip_model_global = self.clip_model_object.model_global
        self.clip_model_local = self.clip_model_object.model_combined
        
        # Separate LoRA layers for global and client-specific
        self.lora_layers_global = self.clip_model_object.lora_layers_global
        self.lora_layers_global_copy = self.clip_model_object.lora_layers_global_copy
        self.lora_layers_local = self.clip_model_object.lora_layers_local
        
        self.logit_scale = self.clip_model_global.state_dict()['logit_scale'].exp()
        
        # Parameters for optimization
        self.global_lora_params = [p for p in self.clip_model_global.parameters() if p.requires_grad]
        self.local_lora_params = [p for p in self.clip_model_local.parameters() if p.requires_grad]
        
        # Optimizers for both global and client-specific LoRA layers
        self.global_optimizer = torch.optim.AdamW(
            params=self.global_lora_params, 
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay
        )
        
        self.mu_global_optimizer = torch.optim.AdamW(
            params=[self.mu_global],
            lr=self.mu_learning_rate_global, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay
        )

        self.local_optimizer = torch.optim.AdamW(
            params=self.local_lora_params, 
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay
        )
        
        self.mu_local_optimizer = torch.optim.AdamW(
            params=[self.mu_local],
            lr=self.mu_learning_rate_local, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay
        )

        self.global_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.global_optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        
        self.local_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.local_optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
            
        trainloader = self.load_train_data()
        self.clip_model_global.to(self.device)
        self.clip_model_local.to(self.device)
        
        train_num = 0
        
        start = time.time()

        for epoch in range(self.local_epochs):
                
            with tqdm(trainloader, total=len(trainloader)) as pbar:  # Initialize pbar here
                for i, batch in enumerate(pbar):

                    images, target, texts = batch
                    
                    images = images.to(self.device)
                    # target = target.to(self.device)
                    texts = texts.to(self.device)

                    # texts is a dictionary, extract the required tensors
                    input_ids = texts['input_ids'].squeeze(1) # Remove the extra dimension
                    attention_mask = texts['attention_mask'].squeeze(1) # Remove the extra dimension
                    
                    
                    # start update local LoRA ------------------------------------------------------------
                    self.clip_model_global.eval()
                    self.clip_model_local.train()
                    
                    # Freeze the global LoRA layers
                    for param in self.lora_layers_global.values():
                        param.requires_grad = False
                    
                    # Obtain features from both global and local models
                    with torch.no_grad():  # No gradient computation for global model
                        image_features_global = self.clip_model_global.get_image_features(images).float()
                        text_features_global = self.clip_model_global.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()
                    image_features_local = self.clip_model_local.get_image_features(images).float()
                    text_features_local = self.clip_model_local.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()

                    # Normalize features
                    image_features_global = image_features_global / image_features_global.norm(dim=1, keepdim=True)
                    text_features_global = text_features_global / text_features_global.norm(dim=1, keepdim=True)
                    image_features_local = image_features_local / image_features_local.norm(dim=1, keepdim=True)
                    text_features_local = text_features_local / text_features_local.norm(dim=1, keepdim=True)
                    
                    # Compute logits for both models
                    logits_per_image_global = self.logit_scale * image_features_global @ text_features_global.t()
                    logits_per_text_global = logits_per_image_global.t()
                    logits_per_image_local = self.logit_scale * image_features_local @ text_features_local.t()
                    logits_per_text_local = logits_per_image_local.t()
                    
                    # Compute cross-entropy loss for client-specific model
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                    cross_entropy_loss = (self.loss(logits_per_image_local, ground_truth) + self.loss(logits_per_text_local, ground_truth)) / 2

                    # Compute KL divergence loss between client and global models
                    kl_div_loss_image = torch.nn.functional.kl_div(
                        logits_per_image_local.log_softmax(dim=-1),
                        logits_per_image_global.softmax(dim=-1),
                        reduction='batchmean'
                    )
                    kl_div_loss_text = torch.nn.functional.kl_div(
                        logits_per_text_local.log_softmax(dim=-1),
                        logits_per_text_global.softmax(dim=-1),
                        reduction='batchmean'
                    )

                    kl_div_loss = (kl_div_loss_image + kl_div_loss_text) / 2
                    
                    # Combine losses
                    total_loss = cross_entropy_loss + kl_div_loss
                    
                    # total_loss = (1 - self.mu_local) * cross_entropy_loss + self.mu_local * kl_div_loss
                    
                    # Optimize Client LoRA
                    self.local_optimizer.zero_grad()
                    # self.mu_local_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_lora_params, max_norm=1.0)
                    self.local_optimizer.step()
                    # self.mu_local_optimizer.step()
                    
                    # Clamp mu_local to the range [0, 1]
                    # with torch.no_grad():
                    #     self.mu_local.clamp_(0, 1)

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Local CE Loss: {cross_entropy_loss.item():.4f}, Local KL Loss: {kl_div_loss.item():.4f}, Local Total Loss: {total_loss.item():.4f}")
                    
                    # Unfreeze the global LoRA layers
                    for param in self.lora_layers_global.values():
                        param.requires_grad = True
                    
                    # end update local LoRA ------------------------------------------------------------
                    
                    
                    # start update global LoRA ------------------------------------------------------------
                    self.clip_model_local.eval()
                    self.clip_model_global.train()
                    
                    # Freeze the local LoRA layers
                    for param in self.lora_layers_local.values():
                        param.requires_grad = False
                        
                    # Obtain features from both global and local models
                    with torch.no_grad():  # No gradient computation for local model
                        image_features_local = self.clip_model_local.get_image_features(images).float()
                        text_features_local = self.clip_model_local.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()
                    
                    image_features_global = self.clip_model_global.get_image_features(images).float()
                    text_features_global = self.clip_model_global.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()

                    # Normalize features
                    image_features_global = image_features_global / image_features_global.norm(dim=1, keepdim=True)
                    text_features_global = text_features_global / text_features_global.norm(dim=1, keepdim=True)
                    image_features_local = image_features_local / image_features_local.norm(dim=1, keepdim=True)
                    text_features_local = text_features_local / text_features_local.norm(dim=1, keepdim=True)
                    
                    # Compute logits for both models
                    logits_per_image_global = self.logit_scale * image_features_global @ text_features_global.t()
                    logits_per_text_global = logits_per_image_global.t()
                    logits_per_image_local = self.logit_scale * image_features_local @ text_features_local.t()
                    logits_per_text_local = logits_per_image_local.t()
                    
                    # Compute cross-entropy loss for global model
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                    cross_entropy_loss = (self.loss(logits_per_image_global, ground_truth) + self.loss(logits_per_text_global, ground_truth)) / 2

                    # Compute KL divergence loss between global and client models
                    kl_div_loss_image = torch.nn.functional.kl_div(
                        logits_per_image_global.log_softmax(dim=-1),
                        logits_per_image_local.softmax(dim=-1),
                        reduction='batchmean'
                    )
                    kl_div_loss_text = torch.nn.functional.kl_div(
                        logits_per_text_global.log_softmax(dim=-1),
                        logits_per_text_local.softmax(dim=-1),
                        reduction='batchmean'
                    )
                    
                    print(f'kl_div_loss_image: {kl_div_loss_image}')
                    print(f'kl_div_loss_text: {kl_div_loss_text}')

                    kl_div_loss = (kl_div_loss_image + kl_div_loss_text) / 2
                    
                    # Combine losses
                    total_loss = cross_entropy_loss + kl_div_loss
                    
                    # total_loss = (1 - self.mu_global) * cross_entropy_loss + self.mu_global * kl_div_loss
                    
                    # Optimize global LoRA
                    self.global_optimizer.zero_grad()
                    # self.mu_global_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_lora_params, max_norm=1.0)
                    self.global_optimizer.step()
                    # self.mu_global_optimizer.step()
                    
                    # Clamp mu_global to the range [0, 1]
                    # with torch.no_grad():
                    #     self.mu_global.clamp_(0, 1)

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Global CE Loss: {cross_entropy_loss.item():.4f}, Global KL Loss: {kl_div_loss.item():.4f}, Global Total Loss: {total_loss.item():.4f}")
                    
                    # Unfreeze the local LoRA layers
                    for param in self.lora_layers_local.values():
                        param.requires_grad = True
                    
                    # end update global LoRA ------------------------------------------------------------
                    
                    
                    train_num += target.size(0)
                    
        self.clip_model_global.to("cpu")            
        self.clip_model_local.to("cpu")
        
        print(f'self.mu_global: {self.mu_global}')
        print(f'self.mu_local: {self.mu_local}')

        end = time.time()
        elapsed = end-start
        # print(f"Number of training samples: {train_num}")
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        if self.learning_rate_decay:
            self.global_learning_rate_scheduler.step()
            self.local_learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        
        
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.clip_model_local.to(self.device)
        self.clip_model_local.eval()
        
        train_num = 0
        total_losses = 0

        with torch.no_grad():
            for i, batch in enumerate(trainloader):

                images, target, texts = batch

                images = images.to(self.device)
                # target = target.to(self.device)
                texts = texts.to(self.device)

                # texts is a dictionary, extract the required tensors
                input_ids = texts['input_ids'].squeeze(1) # Remove the extra dimension
                attention_mask = texts['attention_mask'].squeeze(1) # Remove the extra dimension


                image_features = self.clip_model_local.get_image_features(images).float()

                text_features = self.clip_model_local.get_text_features(input_ids=input_ids, 
                                                            attention_mask=attention_mask).float()


                image_features = image_features / \
                    image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                    text_features.norm(dim=1, keepdim=True)

                # logit_scale = model.model.logit_scale.exp()
                # logit_scale = self.clip_model.state_dict()['logit_scale'].exp()
                logits_per_image = self.logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()


                # Compute loss
                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                loss = (self.loss(logits_per_image, ground_truth) + self.loss(logits_per_text, ground_truth))/2

                # pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                print(f"Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")

                train_num += target.size(0)
                total_losses += loss.item() * target.size(0)
                    
        self.clip_model_local.to("cpu")
        
        print(f"Number of training samples: {train_num}")
        print(f"Total training loss after training: {total_losses:.4f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        
        return total_losses, train_num
    
    def test_metrics(self):
        testloader = self.load_test_data()
        self.clip_model_local.to(self.device)
        self.clip_model_local.eval()
                
        with torch.no_grad():
            top1_1, test_num = 0., 0

            for i, (images, target, texts) in enumerate(testloader):
                images = images.to(self.device)
                target = target.to(self.device)
                texts = texts.to(self.device)

                # predict
                image_features = self.clip_model_local.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # measure accuracy of 1 template
                zeroshot_weights_1 = return_zeroshot_weight(self.dataset, self.clip_model_local, self.processor, self.class_names, self.device)
                logits = self.logit_scale * image_features @ zeroshot_weights_1
                acc1 = accuracy(logits, target, topk=(1,))
                top1_1 += acc1[0]

                test_num += images.size(0)
                
        self.clip_model_local.to("cpu")
        
        # top1_1 = (top1_1 / test_num) 
        print(f"Number of testing samples: {test_num}")
        # print(f"Top-1 accuracy: {top1_1:.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        return top1_1, test_num
    
    def set_parameters(self, dictionary):
        self.clip_model_object.set_lora_dict_global(dictionary)
        
    def set_lora_global_with_momentum(self, dictionary):    
        self.clip_model_object.set_lora_dict_global_with_momentum(dictionary)
        
    def set_lora_local_with_momentum(self, dictionary):    
        self.clip_model_object.set_lora_dict_local_with_momentum(dictionary)
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    home_dir = '/work/LAS/jannesar-lab/dphuong'
    # home_dir = '/scratch/bczq'
    model_checkpoint = "openai/clip-vit-base-patch32"
    
    cache_dir = home_dir / "models"
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    