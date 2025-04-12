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
from torch.nn.functional import cosine_similarity

from pathlib import Path

from flcore.trainmodel.clip_model_drop import *


class clientLORASELECTLEASTWOH(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        
        self.lora_params = args.lora_params
        
        self.k_layers = args.k_layers  # Number of least important layers to exclude from LoRA
        
        # Initialize the CLIP model with LoRA layers (applied conditionally later)
        self.clip_model_object = CLIPModelWithLoRA(
            model_checkpoint=args.model_checkpoint, 
            home_dir=args.home_dir, 
            lora_params=self.lora_params
        )
        
        self.vanilla_clip_model = self.clip_model_object.vanilla_model

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

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )


    def calculate_layer_importance(self, data_loader):
        """Calculate importance scores for CLIP's encoder layers."""
        self.vanilla_clip_model.eval()  # Set the model to evaluation mode

        # Move the model to the correct device
        self.vanilla_clip_model.to(self.device)

        # Dictionary to store importance scores for each encoder layer
        layer_scores = {}

        # Hook function to record input and output tensors
        # def forward_hook(module, input, output):
        def forward_hook(layer_idx):
            def hook(module, input, output):
                if not input or not output:  # Check for empty input/output
                    print(f"Skipping layer {layer_idx}: Empty input or output")
                    return

                input_tensor = input[0] if isinstance(input, tuple) else input
                output_tensor = output[0] if isinstance(output, tuple) else output

                if isinstance(input_tensor, torch.Tensor) and isinstance(output_tensor, torch.Tensor):
                    if input_tensor.size(-1) == output_tensor.size(-1):
                        try:
                            score = 1 - cosine_similarity(input_tensor.detach(), output_tensor.detach(), dim=-1).mean().item()
                            # layer_scores[module] = score  # Direct assignment for layer-level score
                            layer_scores[layer_idx] = score  # Assign score using the sequential index
                            # print(f"Score for layer {module}: {score}")
                        except Exception as e:
                            print(f"Error computing similarity for layer {module}: {e}")
                    else:
                        print(f"Skipping layer {module}: Dimension mismatch (Input: {input_tensor.size()}, Output: {output_tensor.size()})")
                else:
                    print(f"Skipping layer {module}: Non-tensor input/output")
            return hook

        # Initialize hooks list
        hooks = []

        # Determine if hooks should be registered for text and vision encoders
        text_keys = [
            'lora_key_text', 'lora_query_text', 'lora_value_text', 
            'lora_outproj_text', 'lora_mlp_text', 'lora_head_text'
        ]
        vision_keys = [
            'lora_key_vision', 'lora_query_vision', 'lora_value_vision', 
            'lora_outproj_vision', 'lora_mlp_vision', 'lora_head_vision'
        ]

        register_text_hooks = any(self.lora_params.get(key, False) for key in text_keys)
        register_vision_hooks = any(self.lora_params.get(key, False) for key in vision_keys)

        # Register hooks for text encoder layers
        if register_text_hooks:
            for i, layer in enumerate(self.vanilla_clip_model.text_model.encoder.layers):
                layer_scores[f"text_{i}"] = 0  # Initialize score for each text encoder layer
                hooks.append(layer.register_forward_hook(forward_hook(f"text_{i}")))

        # Register hooks for vision encoder layers
        if register_vision_hooks:
            for i, layer in enumerate(self.vanilla_clip_model.vision_model.encoder.layers):
                layer_scores[f"vision_{i}"] = 0  # Initialize score for each vision encoder layer
                hooks.append(layer.register_forward_hook(forward_hook(f"vision_{i}")))

        # Pass data through the model
        with torch.no_grad():
            for images, target, texts in tqdm(data_loader, desc="Calculating Layer Importance"):
                images = images.to(self.device)
                texts = texts.to(self.device)

                # Extract text input tensors
                input_ids = texts['input_ids'].squeeze(1)
                attention_mask = texts['attention_mask'].squeeze(1)

                # Perform forward pass
                _ = self.vanilla_clip_model.get_image_features(images)
                _ = self.vanilla_clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # Remove hooks after importance calculation
        for hook in hooks:
            hook.remove()

        # Move the vanilla model back to the cpu
        self.vanilla_clip_model.to("cpu")

        # Sort layers by importance scores
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        for idx, (layer_idx, score) in enumerate(sorted_layers):
            print(f"Layer {layer_idx}: Score = {score}")
        return sorted_layers

    def prepare_least_important_layers(self, sorted_layers):
        """Prepare the least important LoRA layers to be sent to the server."""
        # Get the indices of the least important layers
        least_important_indices = [layer_idx for layer_idx, _ in sorted_layers[-self.k_layers:]]  # Last k layers

        least_important_lora_layers = {}

        for layer_idx in least_important_indices:
            if "text_" in layer_idx:
                prefix = "text_model.encoder.layers"
                encoder_layer = self.clip_model.text_model.encoder.layers[int(layer_idx.split("_")[1])]
            elif "vision_" in layer_idx:
                prefix = "vision_model.encoder.layers"
                encoder_layer = self.clip_model.vision_model.encoder.layers[int(layer_idx.split("_")[1])]
            else:
                print(f"Unknown layer identifier: {layer_idx}")
                continue

            # Collect LoRA layers for encoder layers
            for submodule_name, submodule in encoder_layer.named_modules():
                if isinstance(submodule, LinearWithLoRA):
                    # Construct full layer names with prefixes
                    full_layer_name = f"{prefix}.{layer_idx.split('_')[1]}.{submodule_name}"
                    least_important_lora_layers[f"{full_layer_name}.W_a"] = submodule.lora.W_a
                    least_important_lora_layers[f"{full_layer_name}.W_b"] = submodule.lora.W_b

        # Handle projection layers
        # if self.lora_params.get('lora_head_text', False):
        #     least_important_lora_layers["text_projection.W_a"] = self.lora_layers['text_projection.W_a']
        #     least_important_lora_layers["text_projection.W_b"] = self.lora_layers['text_projection.W_b']

        # if self.lora_params.get('lora_head_vision', False):
        #     least_important_lora_layers["visual_projection.W_a"] = self.lora_layers['visual_projection.W_a']
        #     least_important_lora_layers["visual_projection.W_b"] = self.lora_layers['visual_projection.W_b']

        return least_important_lora_layers

    def send_least_important_layers(self, sorted_layers):
        """Send only the least important LoRA layers to the server."""
        least_important_layers = self.prepare_least_important_layers(sorted_layers)
        self.lora_layers_select = least_important_layers
        print("Least important LoRA layers sent to server:")
        for key in least_important_layers.keys():
            print(f"{key}")

    def train(self):
        """Train on all LoRA layers."""
        trainloader = self.load_train_data()
        self.clip_model.to(self.device)
        self.clip_model.train()
        
        train_num = 0
        start = time.time()

        # Update local parameters and optimizer
        self.local_params = [p for p in self.clip_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params=self.local_params,
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
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
    
    def set_parameters_select(self, dictionary):
        self.clip_model_object.set_lora_dict_select(dictionary)
    
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    home_dir = '/work/LAS/jannesar-lab/dphuong'
    # home_dir = '/scratch/bczq'
    model_checkpoint = "openai/clip-vit-base-patch32"
    
    cache_dir = home_dir / "models"
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    