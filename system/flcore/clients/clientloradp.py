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
from utils.data_utils import return_zeroshot_weight, accuracy
from torch.utils.data import Subset

from pathlib import Path

from flcore.trainmodel.clip_model import *

from fastDP import PrivacyEngine


class clientLORADP(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        
        self.lora_params = args.lora_params

        self.target_epsilon = args.target_epsilon
        
        self.clip_model_object = CLIPModelWithLoRA(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params=self.lora_params)
        
        self.clip_model = self.clip_model_object.model
        
        self.lora_layers = self.clip_model_object.lora_layers 
        
        self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()
        
        self.local_params = [p for p in self.clip_model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            params=self.local_params, 
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2), 
            eps=self.eps, 
            weight_decay=self.weight_decay
        )

        # privacy_engine = PrivacyEngine(
        #     self.local_params,
        #     # batch_size=256,
        #     batch_size=self.batch_size_train,
        #     # sample_size=50000,
        #     # sample_size=,
        #     # epochs=3,
        #     epochs=self.local_epochs,
        #     target_epsilon=self.target_epsilon,
        #     clipping_fn='automatic',
        #     clipping_mode='MixOpt',
        #     origin_params=None,
        #     clipping_style='all-layer',
        # )

        # privacy_engine.attach(self.optimizer)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

        self.privacy_engine_initialized = False

    def get_privacy_engine_state(self):
        # Manually gather the state of the PrivacyEngine to be saved
        if hasattr(self.clip_model, "privacy_engine"):
            state = {
                "module": self.clip_model.privacy_engine.module,
                "batch_size": self.clip_model.privacy_engine.batch_size,
                "sample_size": self.clip_model.privacy_engine.sample_size,
                "epochs": self.clip_model.privacy_engine.epochs,
                "target_epsilon": self.clip_model.privacy_engine.target_epsilon,
                "clipping_fn": self.clip_model.privacy_engine.clipping_fn,
                "clipping_mode": self.clip_model.privacy_engine.clipping_mode,
                "origin_params": None,
                "clipping_style": self.clip_model.privacy_engine.clipping_style,
            }
            return state
        else:
            print("Warning: PrivacyEngine is not initialized.")
            return None

    def set_privacy_engine_state(self, state):
        # Restore the saved PrivacyEngine state if available
        if state:
            # Check if hooks have already been added to avoid adding them twice
            if hasattr(self.clip_model, "privacy_engine") and self.clip_model.privacy_engine:
                print("PrivacyEngine already has hooks added. Skipping re-addition of hooks.")
                self.privacy_engine_initialized = True
                return

            privacy_engine = PrivacyEngine(
                module=self.clip_model,
                batch_size=self.batch_size_train,
                sample_size=len(self.load_train_data().dataset),
                epochs=self.local_epochs,
                target_epsilon=self.target_epsilon,
                clipping_fn='automatic',
                clipping_mode='MixOpt',
                origin_params=None,
                clipping_style='all-layer',
            )
            privacy_engine.set_state(state)
            privacy_engine.attach(self.optimizer)
            self.clip_model.privacy_engine = privacy_engine
            self.privacy_engine_initialized = True

    def train(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.clip_model.train()

        

        # Create a new PrivacyEngine if not already added
        if not hasattr(self.clip_model, "privacy_engine") or not self.privacy_engine_initialized:
            try:
                privacy_engine = PrivacyEngine(
                    module=self.clip_model,
                    batch_size=self.batch_size_train,
                    sample_size=len(trainloader.dataset),
                    epochs=self.local_epochs,
                    target_epsilon=self.target_epsilon,
                    clipping_fn='automatic',
                    clipping_mode='MixOpt',
                    origin_params=None,
                    clipping_style='all-layer',
                )
                privacy_engine.attach(self.optimizer)

                # Check if private_grad is set for each parameter
                # for name, param in self.clip_model.named_parameters():
                #     if hasattr(param, 'private_grad'):
                #         print(f"{name} has private_grad set.")
                #     else:
                #         print(f"{name} does not have private_grad set.")

                self.privacy_engine_initialized = True

                self.clip_model.privacy_engine = privacy_engine  # Mark that the engine is attached
            except Exception as e:
                print(f"Failed to initialize PrivacyEngine: {e}")
                return  # Exit the training function early if initialization fails

        # if not hasattr(self.clip_model, "privacy_engine"):
        #     privacy_engine = PrivacyEngine(
        #         module=self.clip_model,
        #         batch_size=self.batch_size_train,
        #         sample_size=len(trainloader.dataset),
        #         epochs=self.local_epochs,
        #         target_epsilon=self.target_epsilon,
        #         clipping_fn='automatic',
        #         clipping_mode='MixOpt',
        #         origin_params=None,
        #         clipping_style='all-layer',
        #     )

        # privacy_engine = PrivacyEngine(
        #     # self.local_params,
        #     self.clip_model,
        #     # batch_size=256,
        #     batch_size=self.batch_size_train,
        #     # sample_size=50000,
        #     sample_size=len(trainloader.dataset),
        #     # epochs=3,
        #     epochs=self.local_epochs,
        #     target_epsilon=self.target_epsilon,
        #     clipping_fn='automatic',
        #     clipping_mode='MixOpt',
        #     origin_params=None,
        #     clipping_style='all-layer',
        #     # clipping_style='param-wise',
        #     # clipping_style='layer-wise',
        # )

        # Ensure PrivacyEngine is properly initialized
        if not self.privacy_engine_initialized:
            print("PrivacyEngine is not properly initialized.")
            return
        
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
                    self.optimizer.zero_grad()
                    loss.backward()

                    for name, param in self.clip_model.named_parameters():
                        if 'lora' in name and param.requires_grad:
                            if param.grad is not None:
                                param.private_grad = param.grad.clone()
                            else:
                                print(f"Warning: {name} does not have a valid gradient.")

                    # After setting private_grad for LoRA layers
                    # for name, param in self.clip_model.named_parameters():
                    #     if 'lora' in name:
                    #         if hasattr(param, 'private_grad') and param.private_grad is not None:
                    #             print(f"{name} has private_grad set correctly.")
                    #         else:
                    #             print(f"Warning: {name} does not have private_grad set.")

                    torch.nn.utils.clip_grad_norm_(self.local_params, max_norm=1.0)
                    self.optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                    
                    train_num += target.size(0)
                    
        self.clip_model.to("cpu")

        end = time.time()
        elapsed = end-start
        # print(f"Number of training samples: {train_num}")
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed

    def finetune(self):
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.clip_model.train()
        
        train_num = 0
        
        start = time.time()

        for epoch in range(self.fine_tuning_epoch_new):
                
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
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_params, max_norm=1.0)
                    self.optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")
                    
                    train_num += target.size(0)
                    
        self.clip_model.to("cpu")

        end = time.time()
        elapsed = end-start
        # print(f"Number of training samples: {train_num}")
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

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
                    
        self.clip_model.to("cpu")
        
        print(f"Number of training samples: {train_num}")
        print(f"Total training loss after training: {total_losses:.4f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        
        return total_losses, train_num
    
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
    home_dir = '/work/LAS/jannesar-lab/dphuong'
    # home_dir = '/scratch/bczq'
    model_checkpoint = "openai/clip-vit-base-patch32"
    
    cache_dir = home_dir / "models"
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    