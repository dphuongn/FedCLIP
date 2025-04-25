import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path
import copy
import re

from transformers import CLIPProcessor, CLIPModel

# from utils.data_utils import return_zeroshot_weight

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        if rank > 0:
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else (lambda x: x)
            self.scaling = self.alpha / self.rank
        else:
            self.W_a = nn.Parameter(torch.zeros(in_dim, out_dim), requires_grad=False)
            self.W_b = nn.Parameter(torch.zeros(out_dim, out_dim), requires_grad=False)
            self.dropout = lambda x: x
            self.scaling = 0

    def forward(self, x):
        if self.rank > 0:
            x = self.dropout(x)
            return self.scaling * (x @ self.W_a @ self.W_b)
        return torch.zeros_like(x)


class LinearWithLoRA(nn.Module):
    def __init__(self, 
         linear, 
         rank: int = 0, 
         alpha: int = 1, 
         dropout: float = 0.0, 
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank, 
            alpha, 
            dropout, 
        )
        # gating scalar to turn adapter on/off
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.linear(x)
        if self.lora.rank > 0:
            out = out + self.gate * self.lora(x)
        return out

class CLIPModelWithLoRA(nn.Module):
    def __init__(self, model_checkpoint, home_dir, lora_params, layer_keys=[]):
        super().__init__()
        home = Path(home_dir)
        cache_dir = home / "models"
        self.vanilla_model = CLIPModel.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        # freeze base model
        for p in self.model.parameters(): p.requires_grad = False

        self.lora_params = lora_params
        self.lora_layers = {}     # stores all LoRA weights and gates
        self.wa_layers = {}
        self.wb_layers = {}
        self._apply_lora()

        self.layer_keys = layer_keys
        self.lower_keys, self.higher_keys = self.filter_keys()
        self.base = {k: self.lora_layers[k] for k in self.lower_keys}
        self.head = {k: self.lora_layers[k] for k in self.higher_keys}

    def filter_keys(self):
        layer_names, layer_indices = [], []
        for key in self.layer_keys:
            if key.isdigit(): layer_indices.append(int(key))
            elif '-' in key:
                start, end = map(int, key.split('-'))
                layer_indices.extend(range(start, end + 1))
            else:
                layer_names.append(key)
        idx_str = [str(i) for i in layer_indices]
        lower = [k for k in self.lora_layers if not any(n in k for n in layer_names)
                 and not any(re.search(rf'\.{i}\.', k) for i in idx_str)]
        higher= [k for k in self.lora_layers if k not in lower]
        return lower, higher

    def _apply_lora(self):
        assign = partial(LinearWithLoRA, rank=self.lora_params['rank'],
                         alpha=self.lora_params['alpha'], dropout=self.lora_params['dropout'])
        def assign_and_store(layer, attr, name):
            try:
                layer_wrapped = assign(getattr(layer, attr))
                setattr(layer, attr, layer_wrapped)
                # store weights
                for pname, p in layer_wrapped.lora.named_parameters():
                    full = f"{name}.{pname}"
                    self.lora_layers[full] = p
                    if 'W_a' in pname: self.wa_layers[full] = p
                    if 'W_b' in pname: self.wb_layers[full] = p
                # store gate
                self.lora_layers[f"{name}.gate"] = layer_wrapped.gate
            except AttributeError:
                pass
        # text encoder
        for i, ly in enumerate(self.model.text_model.encoder.layers):
            if self.lora_params.get('lora_key_text'):
                assign_and_store(ly.self_attn, 'k_proj', f"text_model.encoder.layers.{i}.self_attn.k_proj")
                assign_and_store(ly.self_attn, 'q_proj', f"text_model.encoder.layers.{i}.self_attn.q_proj")
                assign_and_store(ly.self_attn, 'v_proj', f"text_model.encoder.layers.{i}.self_attn.v_proj")
                assign_and_store(ly.self_attn, 'out_proj', f"text_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params.get('lora_mlp_text'):
                assign_and_store(ly.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store(ly.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")
        if self.lora_params.get('lora_head_text'):
            proj = assign(self.model.text_projection)
            self.model.text_projection = proj
            for pname, p in proj.lora.named_parameters():
                full = f"text_projection.{pname}"
                self.lora_layers[full] = p
            self.lora_layers['text_projection.gate'] = proj.gate
        # vision encoder
        for i, ly in enumerate(self.model.vision_model.encoder.layers):
            if self.lora_params.get('lora_key_vision'):
                assign_and_store(ly.self_attn, 'k_proj', f"vision_model.encoder.layers.{i}.self_attn.k_proj")
                assign_and_store(ly.self_attn, 'q_proj', f"vision_model.encoder.layers.{i}.self_attn.q_proj")
                assign_and_store(ly.self_attn, 'v_proj', f"vision_model.encoder.layers.{i}.self_attn.v_proj")
                assign_and_store(ly.self_attn, 'out_proj', f"vision_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params.get('lora_mlp_vision'):
                assign_and_store(ly.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store(ly.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")
        if self.lora_params.get('lora_head_vision'):
            proj = assign(self.model.visual_projection)
            self.model.visual_projection = proj
            for pname, p in proj.lora.named_parameters():
                self.lora_layers[f"visual_projection.{pname}"] = p
            self.lora_layers['visual_projection.gate'] = proj.gate

    def set_lora_dict(self, dictionary):
        for key, p in self.lora_layers.items():
            if key not in dictionary:
                raise KeyError(f"Missing key {key}")
            p.data.copy_(dictionary[key].data)

    def print_lora_dict_shapes(self, d):
        for k, p in d.items():
            print(f"{k}: {tuple(p.shape)}")
    
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
        

        'lora_query_vision': True,
        'lora_key_vision': True,
        'lora_value_vision': True,
        'lora_outproj_vision': True,
        'lora_mlp_vision': True,
        'lora_head_vision': True,
    }


    # test CLIPModelWithLoRA
    
    CLIPModelWithLoRA_object = CLIPModelWithLoRA(model_checkpoint=model_checkpoint, home_dir=HOME, lora_params=lora_params).to(device)
    
    model = CLIPModelWithLoRA_object.model
    
    lora_layers = CLIPModelWithLoRA_object.lora_layers
    
    print(f'lora_layers: {lora_layers}')
    
    # CLIPModelWithLoRA_object.print_lora_dict_shapes(lora_layers)
    
    num_lora_params =  CLIPModelWithLoRA_object.count_lora_parameters()
    
    print(f'number of lora params: {num_lora_params:,}')
    
    lora_size = CLIPModelWithLoRA_object.calculate_lora_size()
    
    print(f'lora size: {lora_size:.3f} MB')
    
    layer_index = 1
    
    lora_param_count = CLIPModelWithLoRA_object.count_lora_parameters_layer(layer_index)
    lora_size_mb = CLIPModelWithLoRA_object.calculate_lora_size_layer(layer_index)

    print(f"Number of LoRA parameters in layer {layer_index}: {lora_param_count}")
    print(f"Size of LoRA adapter in layer {layer_index}: {lora_size_mb} MB")

    
    lora_param_count_head = CLIPModelWithLoRA_object.count_lora_parameters_head()
    lora_size_mb_head = CLIPModelWithLoRA_object.calculate_lora_size_head()

    print(f"Number of LoRA parameters in head: {lora_param_count_head}")
    print(f"Size of LoRA adapter in head: {lora_size_mb_head} MB")


    
#     model_size = CLIPModelWithLoRA_object.calculate_model_size()
    
#     print(f'model size: {model_size:.3f} MB')
    
#     state_dict_size = CLIPModelWithLoRA_object.calculate_state_dict_size(lora_layers)
    
#     print(f'state_dict size: {state_dict_size:.3f} MB')
    
    
#     lora_layers_neo = copy.deepcopy(lora_layers)
    
#     print(f'compare 2 dicts before: {CLIPModelWithLoRA_object.compare_lora_dicts(lora_layers, lora_layers_neo)}')
    
    
    
#     W_a = nn.Parameter(torch.zeros(512, 2))
#     W_b = nn.Parameter(torch.ones(2, 512))
    
#     for name, param in lora_layers_neo.items():
#         if 'W_a' in name:
#             param.data.copy_(W_a)
#         elif 'W_b' in name:
#             param.data.copy_(W_b)

    
#     print(f'compare 2 dicts after: {CLIPModelWithLoRA_object.compare_lora_dicts(lora_layers, lora_layers_neo)}')
    
#     # set lora_layers as lora_layers_neo
    
#     CLIPModelWithLoRA_object.set_lora_dict(lora_layers_neo)
    
#     lora_layers2 = CLIPModelWithLoRA_object.lora_layers
    
#     print(f'compare 2 dicts after setting: {CLIPModelWithLoRA_object.compare_lora_dicts(lora_layers_neo, lora_layers2)}')
    
    
    
    # test CLIPModelFFT
    
#     CLIPModelFFT_object = CLIPModelFFT(model_checkpoint=model_checkpoint, home_dir=HOME).to(device)
    
#     CLIPModelFFT_object.print_named_parameters()
    
