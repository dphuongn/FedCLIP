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
from utils.data_utils import return_zeroshot_weight, accuracy
from torch.utils.data import Subset

from pathlib import Path

from flcore.trainmodel.clip_model import *


class clientSORALOCAL(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        
        self.lora_params = args.lora_params
        self.sparse_lambda = args.sparse_lambda
        self.lambda_schedule = args.lambda_schedule
        self.max_lambda = args.max_lambda
        self.lambda_num = args.lambda_num
        
        self.clip_model_object = CLIPModelWithSoRA(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params=self.lora_params)
        
        self.clip_model = self.clip_model_object.model
        
        self.lora_layers = self.clip_model_object.lora_layers 
        
        self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()
        
        # Collect all gate parameters and W_a, W_b separately
        self.gate_params = []
        self.lora_params = []
        
        # self.sora_params = [p for p in self.clip_model.parameters() if p.requires_grad]
        
        for name, module in self.clip_model.named_modules():
            if isinstance(module, SoRALayer):
                # Gate parameters for sparse optimization
                if hasattr(module, 'gate'):
                    self.gate_params.append(module.gate)
                # W_a and W_b parameters for regular optimization
                if hasattr(module, 'W_a'):
                    self.lora_params.append(module.W_a)
                if hasattr(module, 'W_b'):
                    self.lora_params.append(module.W_b)
        
        # Initialize the SparseAdamW optimizer
        self.sparse_optimizer = SparseAdamW(
            sparse_lambda=self.sparse_lambda,  # Sparsity regularization term
            lambda_schedule=self.lambda_schedule,  # Schedule type (linear, log_linear, etc.)
            max_lambda=self.max_lambda,  # Maximum lambda value
            lambda_num=self.lambda_num,  # Number of lambda updates
            # clip_value=args.clip_value  # Optional: gradient clipping value
            params=self.gate_params,
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay,
        )
        
        # Initialize the AdamW optimizer from transformers for W_a and W_b matrices
        self.lora_optimizer = torch.optim.AdamW(
            params=self.lora_params,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        self.learning_rate_scheduler_lora = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.lora_optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.clip_model.train()
        
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


                    image_features = self.clip_model.get_image_features(images).float()

                    text_features = self.clip_model.get_text_features(input_ids=input_ids, 
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
                    
                    # loss = (self.loss(logits_per_image, ground_truth))

                    # Backward pass
                    self.sparse_optimizer.zero_grad()  # Zero out gradients for gate parameters
                    self.lora_optimizer.zero_grad()  # Zero out gradients for W_a and W_b
                    loss.backward()
                    
                    # Optimizer step for sparse gates
                    self.sparse_optimizer.step()
                    # self.sparse_optimizer.step_lambda()  # Update lambda for the sparse gates
                    
                    # Optimizer step for W_a and W_b
                    self.lora_optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                    
                    train_num += target.size(0)
                    
        self.clip_model.to("cpu")

        end = time.time()
        elapsed = end-start
        # print(f"Number of training samples: {train_num}")
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        if self.learning_rate_decay:
            self.learning_rate_scheduler_lora.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
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

                image_features = self.clip_model.get_image_features(images).float()
                text_features = self.clip_model.get_text_features(input_ids=input_ids, 
                                                            attention_mask=attention_mask).float()

                image_features = image_features / \
                    image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                    text_features.norm(dim=1, keepdim=True)

                logits_per_image = self.logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t() # = self.logit_scale * text_features @ image_features.t()

                # Compute loss
                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                loss = (self.loss(logits_per_image, ground_truth) + self.loss(logits_per_text, ground_truth))/2

                # pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                print(f"Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")

                train_num += target.size(0)
                total_losses += loss.item() * target.size(0)
                    
        self.clip_model.to("cpu")
        
        print(f"Number of training samples: {train_num}")
        print(f"Total training loss after training: {total_losses:.4f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        
        return total_losses, train_num
    
    def test_metrics(self):
        testloader = self.load_test_data(id=0)
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
        
        # top1_1 = (top1_1 / test_num) * 100
        print(f"Number of testing samples: {test_num}")
        print(f"Top-1 accuracy: {top1_1:.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        return top1_1, test_num
    
    
    def set_parameters(self, dictionary):
        self.clip_model_object.set_lora_dict(dictionary)
        
    def print_sora_ranks(self):
        print(f"sora rank of client {self.id}:")
        """Print the active rank for each SoRA layer after training."""
        # for layer_name, lora_layer in self.lora_layers.items():
            # if hasattr(lora_layer, 'gate'):
            #     # Count non-zero gates
            #     active_ranks = (lora_layer.gate.abs() > 1e-6).sum().item()
            #     print(f"Layer: {layer_name}, Effective Rank: {active_ranks}/{lora_layer.rank}")
            # Ensure we're working with SoRA layers
        for name, module in self.clip_model.named_modules():
            if isinstance(module, SoRALayer) and hasattr(module, 'gate'):
                gate_values = module.gate.detach().cpu().numpy()
                active_ranks = (abs(gate_values) > 1e-6).sum()  # Count non-zero gates
                total_ranks = module.gate.shape[1]  # Total number of ranks (same as initial rank)
                print(f"Layer: {name}, Effective Rank: {active_ranks}/{total_ranks}")
    
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    