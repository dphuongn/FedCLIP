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

from flcore.trainmodel.clip_model_simple import *


# --- MetaLoRA Client ---
class clientMeta(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        # meta‐hyperparams
        self.inner_steps = args.inner_steps
        self.alpha = args.meta_inner_lr
        # self.meta_lr = args.meta_outer_lr
        self.meta_lr = self.learning_rate
        self.gumbel_temp = args.gumbel_temp
        self.meta_support_fraction = getattr(args, 'meta_support_fraction', 0.5)

        # distillation & consistency weights
        self.kl_gamma            = args.kl_gamma
        self.consistency_lambda  = args.consistency_lambda
        self.sparsity_lambda     = args.sparsity_lambda
        self.distill_temp        = args.distill_temp


        # load a single LoRA‐wrapped CLIP
        self.clip_model_object = CLIPModelWithLoRA(
            model_checkpoint=args.model_checkpoint,
            home_dir=args.home_dir,
            lora_params=args.lora_params
        )
        self.clip_model = self.clip_model_object.model

        # prepare a frozen teacher copy for distillation
        self.teacher_model = copy.deepcopy(self.clip_model).to(self.device).eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        # gather LoRA & gate params
        layers = self.clip_model_object.lora_layers
        self.meta_params = [
            p for n,p in layers.items()
            if n.endswith('.W_a') or n.endswith('.W_b')
        ]
        self.gate_params = [
            p for n,p in layers.items()
            if n.endswith('.gate')
        ]

        # optimizer will update both weights and gates
        self.meta_optimizer = torch.optim.AdamW(
            self.meta_params + self.gate_params,
            lr=self.meta_lr
        )

        # keep the logit_scale handy
        self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()

    def parameters_to_vector(self, params):
        return torch.cat([p.view(-1) for p in params], dim=0)

    def get_meta_split_loaders(self, batch_size=None):
        # 1) determine batch size
        if batch_size is None:
            batch_size = self.batch_size_train

        # 2) load full client dataset
        full_data = read_client_data_clip(
            self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True
        )
        N = len(full_data)

        # 3) record total training samples for this client
        self.train_samples = int(N * self.train_data_fraction)

        # 4) sample exactly train_samples indices without replacement
        train_indices = np.random.choice(N, self.train_samples, replace=False)

        # 5) shuffle those indices and split into support/query
        np.random.shuffle(train_indices)
        support_size = int(self.train_samples * self.meta_support_fraction)
        support_idx = train_indices[:support_size]
        query_idx   = train_indices[support_size:]

        # 6) build DataLoaders
        support_loader = DataLoader(
            Subset(full_data, support_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        query_loader = DataLoader(
            Subset(full_data, query_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        return support_loader, query_loader


    # def get_meta_split_loaders(self):
    #     data = read_client_data_clip(
    #         self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True
    #     )
    #     N = len(data); idx = np.random.permutation(N)
    #     k = int(N * self.meta_support_fraction)
    #     return (
    #         DataLoader(Subset(data, idx[:k]), batch_size=self.batch_size_train, shuffle=True),
    #         DataLoader(Subset(data, idx[k:]), batch_size=self.batch_size_train, shuffle=True)
    #     )

    def sample_and_apply_gates(self):
        if not self.gate_params:
            return
        mask = F.gumbel_softmax(
            torch.stack(self.gate_params),
            tau=self.gumbel_temp,
            hard=True
        )
        for g,m in zip(self.gate_params, mask):
            g.data.copy_(m)

    def _load_meta_vec(self, vec):
        # write a flat vec back into all W_a/W_b in self.clip_model
        state = copy.deepcopy(self.clip_model.state_dict())
        names = [
            n for n in self.clip_model_object.lora_layers
            if n.endswith('.W_a') or n.endswith('.W_b')
        ]
        sizes = [self.clip_model_object.lora_layers[n].numel() for n in names]
        splits = torch.split(vec, sizes)
        for n,chunk in zip(names, splits):
            state[n] = chunk.view(self.clip_model_object.lora_layers[n].shape)
        self.clip_model.load_state_dict(state, strict=False)

    def compute_local_loss(self, images, target, texts):
        # move to GPU
        images, texts = images.to(self.device), texts.to(self.device)
        input_ids    = texts['input_ids'].squeeze(1)
        attention_mask = texts['attention_mask'].squeeze(1)

        # STUDENT forward
        im_s = self.clip_model.get_image_features(images).float()
        tx_s = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        ).float()
        im_s = im_s / im_s.norm(dim=1, keepdim=True)
        tx_s = tx_s / tx_s.norm(dim=1, keepdim=True)
        ls = self.logit_scale
        logits_i_s = ls * im_s @ tx_s.t()
        logits_t_s = logits_i_s.t()

        # TEACHER forward (no grad)
        with torch.no_grad():
            im_t = self.teacher_model.get_image_features(images).float()
            tx_t = self.teacher_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            ).float()
            im_t = im_t / im_t.norm(dim=1, keepdim=True)
            tx_t = tx_t / tx_t.norm(dim=1, keepdim=True)
            logits_i_t = ls * im_t @ tx_t.t()
            logits_t_t = logits_i_t.t()

        # 1) cross‐entropy
        gt = torch.arange(len(images), device=self.device)
        ce = (self.loss(logits_i_s, gt) + self.loss(logits_t_s, gt)) / 2

        # 2) KL‐distillation
        t = self.distill_temp
        p_i = F.log_softmax(logits_i_s / t, dim=-1)
        q_i = F.softmax(     logits_i_t / t, dim=-1)
        kl_i = F.kl_div(p_i, q_i, reduction='batchmean') * (t**2)

        p_t = F.log_softmax(logits_t_s / t, dim=-1)
        q_t = F.softmax(     logits_t_t / t, dim=-1)
        kl_t = F.kl_div(p_t, q_t, reduction='batchmean') * (t**2)

        kl_loss = (kl_i + kl_t) / 2

        # 3) embedding consistency (MSE)
        cons_i = F.mse_loss(im_s, im_t)
        cons_t = F.mse_loss(tx_s, tx_t)
        cons_loss = (cons_i + cons_t) / 2

        # total
        return ce \
             + self.kl_gamma      * kl_loss \
             + self.consistency_lambda * cons_loss

    # def compute_local_loss(self, images, target, texts):
    #     images, texts = images.to(self.device), texts.to(self.device)
    #     ids, att = texts['input_ids'].squeeze(1), texts['attention_mask'].squeeze(1)
    #     imf = self.clip_model.get_image_features(images).float()
    #     txf = self.clip_model.get_text_features(input_ids=ids, attention_mask=att).float()
    #     imf = imf / imf.norm(dim=1, keepdim=True); txf = txf / txf.norm(dim=1, keepdim=True)
    #     ls = self.clip_model.logit_scale.exp()
    #     l_i = ls * imf @ txf.t(); l_t = l_i.t()
    #     gt = torch.arange(len(images), device=self.device)
    #     return (self.loss(l_i,gt) + self.loss(l_t,gt)) / 2

    def _compute_loss(self, loader):
        total = 0.0
        for imgs, tgt, txt in loader:
            total += self.compute_local_loss(imgs, tgt, txt)
        return total / len(loader)

    def send_meta_update(self, update_dict):
        # forward this dict up to your server
        self.server.receive_meta_updates(update_dict)

    def train(self):
        # move LoRA‐CLIP to device
        self.clip_model.to(self.device).train()

        # get support / query
        sup, qry = self.get_meta_split_loaders()

        # flatten meta‐params
        orig    = self.parameters_to_vector(self.meta_params)
        adapted = orig.clone().detach().requires_grad_(True)

        # 1) inner‐loop on support
        for _ in range(self.inner_steps):
            self._load_meta_vec(adapted)
            self.sample_and_apply_gates()
            loss = self._compute_loss(sup)
            grads = torch.autograd.grad(
                loss, [adapted], create_graph=True, allow_unused=True
            )[0]
            if grads is None:
                grads = torch.zeros_like(adapted)
            adapted = adapted - self.alpha * grads

        # 2) outer‐loop on query + sparsity penalty
        self._load_meta_vec(adapted)
        self.sample_and_apply_gates()
        outer = self._compute_loss(qry)

        # L1 on gates
        if self.gate_params:
            gate_vec = torch.stack(self.gate_params)
            outer = outer + self.sparsity_lambda * gate_vec.abs().sum()

        self.meta_optimizer.zero_grad()
        outer.backward()
        self.meta_optimizer.step()

        # send updated meta‐params back
        updated = {
            n: p.data.clone()
            for n,p in self.clip_model_object.lora_layers.items()
        }
        self.send_meta_update(updated)

        # move back to CPU
        self.clip_model.to("cpu")

    # def train(self):
    #     # Ensure model on correct device
    #     self.clip_model.to(self.device)
    #     self.clip_model.train()

    #     sup, qry = self.get_meta_split_loaders()
    #     orig = self.parameters_to_vector(self.meta_params)
    #     adapted = orig.clone().detach().requires_grad_(True)
    #     for _ in range(self.inner_steps):
    #         self._load_meta_vec(adapted)
    #         self.sample_and_apply_gates()
    #         loss = self._compute_loss(sup)
    #         grads = torch.autograd.grad(loss, [adapted], create_graph=True, allow_unused=True)[0]
    #         if grads is None: grads = torch.zeros_like(adapted)
    #         adapted = adapted - self.alpha * grads
    #     self._load_meta_vec(adapted)
    #     self.sample_and_apply_gates()
    #     outer = self._compute_loss(qry)
    #     self.meta_optimizer.zero_grad()
    #     outer.backward()
    #     self.meta_optimizer.step()

    #     # self.send_meta_update({n: p.data.clone() for n, p in self.clip_model_object.lora_layers.items()})

    #     updated = {
    #         n: p.data.clone()
    #         for n, p in self.clip_model_object.lora_layers.items()
    #     }
    #     self.send_meta_update(updated)
    #     self.clip_model.to("cpu")

    def test_metrics(self):
        """
        Evaluate the adapted CLIP + LoRA model on the local test split, using the accuracy util.
        """
        loader = self.load_test_data()
        self.clip_model.to(self.device).eval()
        total, top1_count = 0, 0.0
        with torch.no_grad():
            for images, target, texts in loader:
                images, target = images.to(self.device), target.to(self.device)
                # 1) extract image features
                imf = self.clip_model.get_image_features(images).float()
                imf = imf / imf.norm(dim=-1, keepdim=True)
                # 2) zero-shot weights
                zsw = return_zeroshot_weight(
                    self.dataset, self.clip_model, self.processor, self.class_names, self.device
                )
                # 3) compute logits
                logits = self.clip_model.logit_scale.exp() * imf @ zsw
                # 4) accumulate correct predictions via util
                batch_top1 = accuracy(logits, target, topk=(1,))[0]
                top1_count += batch_top1
                total += images.size(0)
        self.clip_model.to("cpu")
        # compute percentage
        # acc1 = top1_count / total * 100
        print(f"Number of testing samples: {total}")
        print(f"Top-1 count: {top1_count:.2f}%")
        print(f"Memory used: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
        return top1_count, total

    def set_parameters(self, dictionary):
        self.clip_model_object.set_lora_dict(dictionary)

    

# # --- Meta-Learning–Enhanced Adaptive LoRA Client ---
# class clientMeta(Client):
#     def __init__(self, args, id, **kwargs):
#         super().__init__(args, id, **kwargs)
#         # Meta-learning hyperparameters
#         self.inner_steps = args.inner_steps             # number of local adaptation steps
#         self.alpha = args.meta_inner_lr                  # inner-loop lr
#         self.meta_lr = args.meta_outer_lr                # outer-loop lr
#         self.gumbel_temp = args.gumbel_temp              # for gating
#         self.meta_support_fraction = getattr(args, 'meta_support_fraction', 0.5)

#         self.lora_params = args.lora_params
#         self.clip_model_object = CLIPModelWithLoRA(model_checkpoint=args.model_checkpoint, home_dir=args.home_dir, lora_params=self.lora_params)
        
#         self.clip_model = self.clip_model_object.model
        
#         self.lora_layers = self.clip_model_object.lora_layers 

#         # Extract parameter lists from the model's lora_layers_local dict
#         layers = self.lora_layers

#         # print(f'layers: {layers}')

#         # LoRA weights (W_a and W_b)
#         self.meta_params = [p for name, p in layers.items() if name.endswith('.W_a') or name.endswith('.W_b')]

#         # print(f'self.meta_params: {self.meta_params}')

#         # gating scalars
#         self.gate_params = [p for name, p in layers.items() if name.endswith('.gate')]

#         # print(f'self.gate_params: {self.gate_params}')
        
#         self.logit_scale = self.clip_model.state_dict()['logit_scale'].exp()


#         # Meta-optimizer over both weights and gates
#         self.meta_optimizer = AdamW(self.meta_params + self.gate_params, lr=self.meta_lr)
    
#     def parameters_to_vector(self, params):
#         """
#         Flatten a list of parameter tensors into a single 1D vector.
#         """
#         return torch.cat([p.view(-1) for p in params], dim=0)

#     def get_meta_split_loaders(self):
#         """
#         Split the client's training data into support and query loaders for meta-learning.
#         """
#         # Load full client dataset
#         full_data = read_client_data_clip(
#             self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True
#         )
#         total = len(full_data)
#         support_size = int(total * self.meta_support_fraction)
#         # Randomly shuffle indices
#         indices = np.random.permutation(total)
#         support_idx = indices[:support_size]
#         query_idx = indices[support_size:]

#         support_ds = Subset(full_data, support_idx)
#         query_ds = Subset(full_data, query_idx)

#         support_loader = DataLoader(
#             support_ds,
#             batch_size=self.batch_size_train,
#             shuffle=True,
#             drop_last=False
#         )
#         query_loader = DataLoader(
#             query_ds,
#             batch_size=self.batch_size_train,
#             shuffle=True,
#             drop_last=False
#         )
#         return support_loader, query_loader

#     def sample_and_apply_gates(self):
#         # sample mask for each gate param via Gumbel-Softmax
#         # flatten gates into vector, sample, then unflatten
#         gate_tensor = torch.stack(self.gate_params)
#         mask = F.gumbel_softmax(gate_tensor, tau=self.gumbel_temp, hard=True)
#         # apply mask in-place
#         for g, m in zip(self.gate_params, mask):
#             g.data.copy_(m)

#     def train(self):
#         # split data into support and query loaders
#         support_loader, query_loader = self.get_meta_split_loaders()
#         # Save original weights
#         orig_vec = self.parameters_to_vector(self.meta_params)
#         adapted = orig_vec.clone()

#         self.clip_model.to(self.device)
#         self.clip_model.train()

#         # Inner-loop adaptation on support
#         for _ in range(self.inner_steps):
#             # load adapted weights into model
#             self._load_meta_vec(adapted)
#             self.sample_and_apply_gates()
#             loss = self._compute_loss(support_loader)
#             grads = torch.autograd.grad(loss, [adapted], create_graph=True)[0]
#             adapted = adapted - self.alpha * grads

#         # Outer-loop on query
#         self._load_meta_vec(adapted)
#         self.sample_and_apply_gates()
#         outer_loss = self._compute_loss(query_loader)
#         # Meta-update
#         self.meta_optimizer.zero_grad()
#         outer_loss.backward()
#         self.meta_optimizer.step()

#         # send updated weights back to server
#         # updated = {name: p.data.clone() for name, p in layers.items()}
#         # self.send_meta_update(updated)

#         self.clip_model.to("cpu")

#     def _load_meta_vec(self, vec):
#         # load a flat vector into the model's local LoRA W_a and W_b
#         state = copy.deepcopy(self.clip_model_object.model.state_dict())
#         names = [n for n in self.lora_layers if n.endswith('.W_a') or n.endswith('.W_b')]
#         splits = torch.split(vec, [self.lora_layers[n].numel() for n in names])
#         for n, chunk in zip(names, splits):
#             state[n] = chunk.view(self.lora_layers[n].shape)
#         self.clip_model_object.model.load_state_dict(state, strict=False)

#     def _compute_loss(self, loader):
#         # same CE + KL + any consistency
#         total = 0.0
#         for imgs, tgt, txt in loader:
#             total += self.compute_local_loss(imgs, tgt, txt)
#         return total / len(loader)


#     # --- add compute_local_loss using vanilla LoRA finetune logic ---
#     def compute_local_loss(self, images, target, texts):
#         """
#         Compute standard CE loss for a batch using the client's combined LoRA CLIP model.
#         """
#         # Move inputs to device
#         images = images.to(self.device)
#         texts = texts.to(self.device)
#         # Extract tokens
#         input_ids = texts['input_ids'].squeeze(1)
#         attention_mask = texts['attention_mask'].squeeze(1)
#         # Forward through adapted CLIP model
#         # model = self.clip_model_object.model_combined
#         model = self.clip_model
#         image_feats = model.get_image_features(images).float()
#         text_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).float()
#         # Normalize
#         image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
#         text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
#         # Compute logits
#         logit_scale = model.logit_scale.exp()
#         logits_per_image = logit_scale * image_feats @ text_feats.t()
#         logits_per_text = logits_per_image.t()
#         # Ground truth labels
#         gt = torch.arange(len(images), dtype=torch.long, device=self.device)
#         # Cross-entropy loss averaged over both directions
#         return (self.loss(logits_per_image, gt) + self.loss(logits_per_text, gt)) / 2


#     def test_metrics(self):
#         testloader = self.load_test_data()
#         # model = self.clip_model_object.model_combined
#         # model.to(self.device).eval()

#         self.clip_model.to(self.device)
#         self.clip_model.eval()
                
#         # correct = 0
#         total = 0

#         with torch.no_grad():
#             top1_1 = 0.

#             for images, target, texts in testloader:
#                 images = images.to(self.device)
#                 target = target.to(self.device)

#                 # 1) image features
#                 image_feats = self.clip_model.get_image_features(images).float()
#                 image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

#                 # 2) zero‐shot text weights from the adapted model
#                 zeroshot_weights = return_zeroshot_weight(
#                     self.dataset, model, self.processor, self.class_names, self.device
#                 )  # shape: [num_classes, feature_dim]

#                 # 3) logits and prediction
#                 logits = self.logit_scale * image_feats @ zeroshot_weights
#                 preds = logits.argmax(dim=-1)

#                 # 4) accumulate
#                 acc1 = accuracy(logits, target, topk=(1,))
#                 top1_1 += acc1[0]

#                 # correct += (preds == target).sum().item()
#                 total += images.size(0)

#         self.clip_model.to("cpu")
#         acc1 = correct / total * 100
#         print(f"Number of testing samples: {total}")
#         print(f"Top-1 accuracy: {acc1:.2f}%")
#         print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
#         return acc1, total
    
#     def set_parameters(self, dictionary):
#         self.clip_model_object.set_lora_dict(dictionary)
        
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = Path.cwd()
    print("Current Working Directory:", current_directory)
    