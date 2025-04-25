import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path
import copy
import re

from transformers import CLIPProcessor, CLIPModel

# -- Dual-LoRA Modules ------------------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        if rank > 0:
            std = 1 / torch.sqrt(torch.tensor(rank, dtype=torch.float32))
            self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std)
            self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.scaling = alpha / rank
        else:
            self.W_a = nn.Parameter(torch.zeros(in_dim, out_dim), requires_grad=False)
            self.W_b = nn.Parameter(torch.zeros(out_dim, out_dim), requires_grad=False)
            self.dropout = nn.Identity()
            self.scaling = 0.0
    def forward(self, x):
        if self.rank > 0:
            x = self.dropout(x)
            return self.scaling * (x @ self.W_a @ self.W_b)
        return torch.zeros_like(x)

class LinearWithDualLoRA(nn.Module):
    def __init__(self, 
         linear, 
         rank_global: int = 0, 
         alpha_global: int = 1, 
         rank_local: int = 0,
         alpha_local: int = 1,
         dropout: float = 0.0, 
         gate_hidden: int = 64,    # you forgot to pass this in your partial
    ):
    # def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        # two adapters:
        self.lora_global = LoRALayer(linear.in_features, linear.out_features, rank_global, alpha_global, dropout)
        self.lora_local  = LoRALayer(linear.in_features, linear.out_features, rank_local, alpha_local, dropout)
        # gating parameters
        # self.gate_g = nn.Parameter(torch.tensor(1.0))
        # self.gate_l = nn.Parameter(torch.tensor(1.0))
        # self.gate = nn.Parameter(torch.tensor(1.0))
        # replace scalar gate → tiny MLP gating on the *instance* embedding
        self.gating_net = nn.Sequential(
            nn.Linear(linear.in_features, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )

        # ← new: default to normal operation
        self.global_only = False

    def forward(self, x):
        """
        x:        [B, in_features]  — the usual input to this linear
        inst_emb: [B, D]            — the frozen CLIP embedding fθ(x)
        """
        # Base output from the frozen CLIP linear
        base = self.linear(x)
        if self.global_only:
            # Only the global adapter
            return base + self.lora_global(x)
        # Compute gating scores per instance (or per token) without flattening
        g = self.gating_net(x)  # Shape: [B, D_in] -> [B, 1] or [B, L, 1]
        # Broadcast across output dims automatically
        return base + g * self.lora_global(x) + (1 - g) * self.lora_local(x)

# -- Dual-LoRA CLIP wrapper ------------------------------------------------
class CLIPModelWithDualLoRA(nn.Module):
    def __init__(self, checkpoint: str, home_dir: Path, lora_params_global: dict, lora_params_local: dict):
        super().__init__()
        home = Path(home_dir)
        cache_dir = home / "models"
        self.model_checkpoint = checkpoint
        self.lora_params_global = lora_params_global
        self.lora_params_local = lora_params_local

        # self.vanilla_model = CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        self.model_combined = CLIPModel.from_pretrained(self.model_checkpoint, cache_dir=cache_dir)
        # freeze vanilla model
        for p in self.model_combined.parameters(): p.requires_grad = False

        self.lora_layers_global = {}
        self.lora_layers_local = {}
        self.lora_gating       = {}

        # Apply both global copy and local LoRA to combined model
        self._apply_lora_combined()

        

    def _apply_lora_combined(self):
        """
        Apply both global and local LoRA to the model_combined.
        """
        assign = partial(
            LinearWithDualLoRA,  # This will apply the global and local LoRA
            rank_global=self.lora_params_global['rank'],
            alpha_global=self.lora_params_global['alpha'],
            rank_local=self.lora_params_local['rank'],
            alpha_local=self.lora_params_local['alpha'],
            dropout=self.lora_params_global['dropout']
        )

        def patch(layer, attr, name_prefix):
            orig = getattr(layer, attr)
            lora_mod = assign(orig)
            setattr(layer, attr, lora_mod)
            # store global & local
            for pn, p in lora_mod.lora_global.named_parameters():
                self.lora_layers_global[f"{name_prefix}.{pn}"] = p
            for pn, p in lora_mod.lora_local.named_parameters():
                self.lora_layers_local[f"{name_prefix}.{pn}"] = p
            # store gating nets
            for pn, p in lora_mod.gating_net.named_parameters():
                self.lora_gating[f"{name_prefix}.gating_net.{pn}"] = p
        

        def assign_and_store_lora_combined(layer, attr, layer_name):
            try:
                lora_layer_combined = assign_lora_combined(getattr(layer, attr))
                setattr(layer, attr, lora_layer_combined)

                # Store global LoRA parameters
                for param_name, param in lora_layer_combined.lora_global.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers_global[full_param_name] = param

                # Store local LoRA parameters
                for param_name, param in lora_layer_combined.lora_local.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers_local[full_param_name] = param

            except AttributeError as e:
                print(f"Error: {e}")
                print(f"Available attributes in the layer: {dir(layer)}")

        # text encoder
        for i, layer in enumerate(self.model_combined.text_model.encoder.layers):
            base = f"text_model.encoder.layers.{i}.self_attn"
            if self.lora_params_global.get('lora_query_text'):
                patch(layer.self_attn, 'q_proj', f"{base}.q_proj")
            if self.lora_params_global.get('lora_key_text'):
                patch(layer.self_attn, 'k_proj', f"{base}.k_proj")
            if self.lora_params_global.get('lora_value_text'):
                patch(layer.self_attn, 'v_proj', f"{base}.v_proj")
            if self.lora_params_global.get('lora_outproj_text'):
                patch(layer.self_attn, 'out_proj', f"{base}.out_proj")
            if self.lora_params_global.get('lora_mlp_text'):
                patch(layer.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
                patch(layer.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")
        if self.lora_params_global.get('lora_head_text'):
            patch(self.model_combined, 'text_projection', 'text_projection')

        # vision encoder
        for i, layer in enumerate(self.model_combined.vision_model.encoder.layers):
            base = f"vision_model.encoder.layers.{i}.self_attn"
            if self.lora_params_global.get('lora_query_vision'):
                patch(layer.self_attn, 'q_proj', f"{base}.q_proj")
            if self.lora_params_global.get('lora_key_vision'):
                patch(layer.self_attn, 'k_proj', f"{base}.k_proj")
            if self.lora_params_global.get('lora_value_vision'):
                patch(layer.self_attn, 'v_proj', f"{base}.v_proj")
            if self.lora_params_global.get('lora_outproj_vision'):
                patch(layer.self_attn, 'out_proj', f"{base}.out_proj")
            if self.lora_params_global.get('lora_mlp_vision'):
                patch(layer.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
                patch(layer.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")
        if self.lora_params_global.get('lora_head_vision'):
            patch(self.model_combined, 'visual_projection', 'visual_projection')

        # # Apply combined LoRA (both global and local) to the model's text and vision encoders
        # for i, layer in enumerate(self.model_combined.text_model.encoder.layers):
        #     if self.lora_params_global.get('lora_key_text', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'k_proj', f"text_model.encoder.layers.{i}.self_attn.k_proj")
        #     if self.lora_params_global.get('lora_query_text', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'q_proj', f"text_model.encoder.layers.{i}.self_attn.q_proj")
        #     if self.lora_params_global.get('lora_value_text', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'v_proj', f"text_model.encoder.layers.{i}.self_attn.v_proj")
        #     if self.lora_params_global.get('lora_outproj_text', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'out_proj', f"text_model.encoder.layers.{i}.self_attn.out_proj")
        #     if self.lora_params_global.get('lora_mlp_text', False):
        #         assign_and_store_lora_combined(layer.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
        #         assign_and_store_lora_combined(layer.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")

        # if self.lora_params_global.get('lora_head_text', False):
        #     self.model_combined.text_projection = assign_lora_combined(self.model_combined.text_projection)
        #     for param_name, param in self.model_combined.text_projection.lora_global.named_parameters():
        #         self.lora_layers_global[f'text_projection.{param_name}'] = param
        #     for param_name, param in self.model_combined.text_projection.lora_local.named_parameters():
        #         self.lora_layers_local[f'text_projection.{param_name}'] = param

        # # Apply combined LoRA to the vision encoder
        # for i, layer in enumerate(self.model_combined.vision_model.encoder.layers):
        #     if self.lora_params_global.get('lora_key_vision', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'k_proj', f"vision_model.encoder.layers.{i}.self_attn.k_proj")
        #     if self.lora_params_global.get('lora_query_vision', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'q_proj', f"vision_model.encoder.layers.{i}.self_attn.q_proj")
        #     if self.lora_params_global.get('lora_value_vision', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'v_proj', f"vision_model.encoder.layers.{i}.self_attn.v_proj")
        #     if self.lora_params_global.get('lora_outproj_vision', False):
        #         assign_and_store_lora_combined(layer.self_attn, 'out_proj', f"vision_model.encoder.layers.{i}.self_attn.out_proj")
        #     if self.lora_params_global.get('lora_mlp_vision', False):
        #         assign_and_store_lora_combined(layer.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
        #         assign_and_store_lora_combined(layer.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")

        # if self.lora_params_global.get('lora_head_vision', False):
        #     self.model_combined.visual_projection = assign_lora_combined(self.model_combined.visual_projection)
        #     for param_name, param in self.model_combined.visual_projection.lora_global.named_parameters():
        #         self.lora_layers_global[f'visual_projection.{param_name}'] = param
        #     for param_name, param in self.model_combined.visual_projection.lora_local.named_parameters():
        #         self.lora_layers_local[f'visual_projection.{param_name}'] = param

    def set_global_adapter(self, adapter_dict: dict[str, nn.Parameter]):
        """
        Overwrite only the global-LoRA weights from adapter_dict.
        """
        for name, param in self.lora_layers_global.items():
            param.data.copy_(adapter_dict[name].data)

    def freeze_backbone_and_local(self):
        # 1) freeze every CLIP backbone param
        # for p in self.model_combined.parameters():
        #     p.requires_grad = False

        # 2) freeze ALL local‐LoRA params
        for p in self.lora_layers_local.values():
            p.requires_grad = False

        # 3) freeze gating nets
        for _, gate in self.lora_gating.items():    
            gate.requires_grad = False

        # leave lora_layers_global as trainable
    
    def unfreeze_local_and_gate(self):
        # undo the above for local adapters
        for p in self.lora_layers_local.values():
            p.requires_grad = True

        # undo the above for gating nets
        for _, gate in self.lora_gating.items():    
            gate.requires_grad = True

        # backbone stays frozen—clients will re-spawn their own wrapper anyway
    
    def set_global_only(self, enabled: bool):
        """
        Tell every LinearWithDualLoRA in the model to
        ignore local‐LoRA & gating when `enabled=True`.
        """
        for module in self.model_combined.modules():
            if isinstance(module, LinearWithDualLoRA):
                module.global_only = enabled
    
    def set_lora_dict(self, dictionary):
        """
        Set the parameters of the LoRA layers from a dictionary.

        Args:
            dictionary (dict): A dictionary containing parameters for the LoRA layers.

        Others:
            dictionary structure: 
            {'text_model.encoder.layers.0.self_attn.k_proj.W_a': tensor(...),
            'text_model.encoder.layers.0.self_attn.k_proj.W_b': tensor(...),
            'text_model.encoder.layers.1.self_attn.out_proj.W_a': tensor(...),
            'text_model.encoder.layers.1.self_attn.out_proj.W_b': tensor(...),
            'text_model.encoder.layers.2.mlp.fc1.W_a': tensor(...),
            'text_model.encoder.layers.2.mlp.fc1.W_b': tensor(...),
            'text_projection.W_a': tensor(...),
            'text_projection.W_b': tensor(...),
            ...}
        """
        for key, param in self.lora_layers_global.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            param.data.copy_(dictionary[key].data)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    HOME = '/work/LAS/jannesar-lab/dphuong'
    # HOME = "/scratch/bczq/"
    model_checkpoint = "openai/clip-vit-base-patch32"
    
    dataset = 'flowers'   
        
    class_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
    
    lora_params = {
        'rank': 2,
        'alpha': 16,
        'dropout': 0.05,

        'lora_query_text': True,
        'lora_key_text': True,
        'lora_value_text': True,
        'lora_outproj_text': True,
        'lora_mlp_text': True,
        'lora_head_text': True,
        

        # 'lora_query_vision': True,
        # 'lora_key_vision': True,
        # 'lora_value_vision': True,
        # 'lora_outproj_vision': True,
        # 'lora_mlp_vision': True,
        # 'lora_head_vision': True,
    }


    # test CLIPModelWithLoRA
    
    CLIPModelWithDualLoRA_object = CLIPModelWithDualLoRA(checkpoint=model_checkpoint, home_dir=HOME, lora_params_global=lora_params, lora_params_local=lora_params).to(device)
    
    # model = CLIPModelWithDualLoRA_object.model
    
    global_adapters = CLIPModelWithDualLoRA_object.lora_layers_global

    local_adapters = CLIPModelWithDualLoRA_object.lora_layers_local
    
    print(f'global_adapters: {global_adapters}')

    print(f'local_adapters: {local_adapters}')
    
