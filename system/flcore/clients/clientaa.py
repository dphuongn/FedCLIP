import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from transformers import AdamW

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import return_zeroshot_weight, return_zeroshot_weight_aa, accuracy
from torch.utils.data import Subset

from pathlib import Path

from flcore.trainmodel.clip_model import *

import wandb  # Import wandb


class clientAA(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        
        self.aa_params = args.aa_params
        
        self.clip_model_object = CLIPModelWithAttentionAdapter(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, aa_params=self.aa_params).to(args.device)
        
        self.clip_model = self.clip_model_object.model
        
        self.aa = self.clip_model_object.aa
        
        self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()
        
        self.local_params = [p for p in self.aa.parameters() if p.requires_grad]
        
        # self.optimizer = torch.optim.Adam(
        #     self.local_params, 
        #     lr=self.learning_rate, 
        #     betas=(self.beta1, self.beta2), 
        #     eps=self.eps, 
        #     weight_decay=self.weight_decay
        # )

        self.optimizer = AdamW(
            params=self.local_params,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.aa.to(self.device)
        self.clip_model.train()
        self.aa.train()
        
        train_num = 0
        total_losses = 0
        
        start = time.time()

        for epoch in range(self.local_epochs):
                
            with tqdm(trainloader, total=len(trainloader)) as pbar:  # Initialize pbar here
                for batch in pbar:      

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
                    
                    if self.aa_params['aa_vision']:
                    
                        # added adapter---------------------------------------------
                        image_features_att = self.aa(image_features)
                        image_features = torch.mul(image_features_att, image_features)
                        #-----------------------------------------------------------
                        
                    elif self.aa_params['aa_text']:
                        
                        # added adapter---------------------------------------------
                        text_features_att = self.aa(text_features)
                        text_features = torch.mul(text_features_att, text_features)
                        #-----------------------------------------------------------
                    
                    else:
                        raise NotImplementedError 


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

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_description(f"Client {self.id}: Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                    
                    # Log metrics to wandb
                    # wandb.log({"client_id": self.id, "epoch": epoch, "loss": loss.item()})
                    
                    train_num += target.size(0)
                    
        self.clip_model.to("cpu")
        self.aa.to("cpu")

        end = time.time()
        elapsed = end-start
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        
    def finetune(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.aa.to(self.device)
        self.clip_model.train()
        self.aa.train()
        
        train_num = 0
        total_losses = 0
        
        start = time.time()

        for epoch in range(self.fine_tuning_epoch_new):
                
            with tqdm(trainloader, total=len(trainloader)) as pbar:  # Initialize pbar here
                for batch in pbar:      

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
                    
                    if self.aa_params['aa_vision']:
                    
                        # added adapter---------------------------------------------
                        image_features_att = self.aa(image_features)
                        image_features = torch.mul(image_features_att, image_features)
                        #-----------------------------------------------------------
                        
                    elif self.aa_params['aa_text']:
                        
                        # added adapter---------------------------------------------
                        text_features_att = self.aa(text_features)
                        text_features = torch.mul(text_features_att, text_features)
                        #-----------------------------------------------------------
                    
                    else:
                        raise NotImplementedError 


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

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                    
                    train_num += target.size(0)
                    
        self.clip_model.to("cpu")
        self.aa.to("cpu")

        end = time.time()
        elapsed = end-start
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.aa.to(self.device)
        self.clip_model.eval()
        self.aa.eval()
        
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
                
                
                if self.aa_params['aa_vision']:
                    
                    # added adapter---------------------------------------------
                    image_features_att = self.aa(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    #-----------------------------------------------------------
                        
                elif self.aa_params['aa_text']:

                    # added adapter---------------------------------------------
                    text_features_att = self.aa(text_features)
                    text_features = torch.mul(text_features_att, text_features)
                    #-----------------------------------------------------------

                else:
                    raise NotImplementedError 


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
                # print(f"Batch {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}")

                train_num += target.size(0)
                total_losses += loss.item() * target.size(0)
                    
        self.clip_model.to("cpu")
        
        print(f"Number of training samples: {train_num}")
        print(f"Total training loss after training: {total_losses:.4f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        
        return total_losses, train_num
    
    
    def test_metrics(self):
        testloader = self.load_test_data()
        self.clip_model.to(self.device)
        self.aa.to(self.device)
        self.clip_model.eval()
        self.aa.eval()
                
        with torch.no_grad():
            top1_1, test_num = 0., 0

            for i, (images, target, texts) in enumerate(testloader):
                images = images.to(self.device)
                target = target.to(self.device)
                texts = texts.to(self.device)

                # predict
                image_features = self.clip_model.get_image_features(images)
                
                if self.aa_params['aa_vision']:
                    
                    # added adapter---------------------------------------------
                    image_features_att = self.aa(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    #-----------------------------------------------------------
                
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # measure accuracy of 1 template
                if self.aa_params['aa_text']:
                    zeroshot_weights_1 = return_zeroshot_weight_aa(self.dataset, self.clip_model, self.processor, self.class_names, self.device, self.aa)
                else:
                    zeroshot_weights_1 = return_zeroshot_weight(self.dataset, self.clip_model, self.processor, self.class_names, self.device)
                logits = self.logit_scale * image_features @ zeroshot_weights_1
                acc1 = accuracy(logits, target, topk=(1,))
                top1_1 += acc1[0]

                test_num += images.size(0)
                
        self.clip_model.to("cpu")

        print(f"Number of testing samples: {test_num}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        return top1_1, test_num
    
    def set_parameters(self, model):
        self.clip_model_object.set_aa_parameters(model)
    
    
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    
    
#     client = clientAVGC(, 
#                             id=i, 
#                             train_samples=len(train_data), 
#                             test_samples=len(test_data), 
#                             train_slow=train_slow, 
#                             send_slow=send_slow)
#             self.clients.append(client)
    
    
    