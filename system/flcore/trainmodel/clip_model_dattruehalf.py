import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path
import copy
import re

from transformers import CLIPProcessor, CLIPModel

# from utils.data_utils import return_zeroshot_weight


def get_processor(model_checkpoint, home_dir):
    """
    Get the processor for the specified model checkpoint.

    Args:
        model_checkpoint (str): Identifier for the pre-trained model.
        home_dir (str): Directory path for model and processor caching.

    Returns:
        CLIPProcessor: The processor for the specified model.
    """
    home_dir = Path(home_dir)
    cache_dir = home_dir / "models"
    processor = CLIPProcessor.from_pretrained(model_checkpoint, cache_dir=cache_dir)
    return processor

class LoRALayer(nn.Module):
    def __init__(self, 
         in_dim, 
         out_dim, 
         rank: int, 
         alpha: int, 
         dropout: float, 
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        # self.W_b = nn.Parameter(torch.zeros(rank, out_dim))

        # # Dropout
        # self.dropout = nn.Dropout(dropout) if dropout > 0.0 else (lambda x: x)

        # # Scaling
        # # self.scaling = self.alpha / self.rank
        # self.scaling = self.alpha


        if rank > 0:
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else (lambda x: x)
            self.scaling = self.alpha / self.rank
        else:
            # Initialize dummy parameters to avoid errors during training
            self.W_a = nn.Parameter(torch.zeros(in_dim, out_dim), requires_grad=False)
            self.W_b = nn.Parameter(torch.zeros(out_dim, out_dim), requires_grad=False)
            self.dropout = lambda x: x
            self.scaling = 0

        # Mark the LoRA parameters as having private gradients
        self.W_a.private_grad = None
        self.W_b.private_grad = None

    def forward(self, x):
        # if self.rank > 0:
        #     x = self.dropout(x)
        #     x = self.scaling * (x @ self.W_a @ self.W_b)
        # return x

        if self.rank > 0:
            x = self.dropout(x)
            x = self.scaling * (x @ self.W_a @ self.W_b)
        else:
            x = torch.zeros_like(x)  # Ensure no contribution if rank = 0
        return x

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

    def forward(self, x):
        if self.lora.rank > 0:
            return self.linear(x) + self.lora(x)
        else:
            return self.linear(x)  # No contribution from LoRA if rank = 0
        # return self.linear(x) + self.lora(x)

class LinearWithLoRACombined(nn.Module):
    def __init__(self, 
         linear, 
         rank_global: int = 0, 
         alpha_global: int = 1, 
         rank_local: int = 0,
         alpha_local: int = 1,
         dropout: float = 0.0, 
    ):
        super().__init__()
        self.linear = linear
        
        # Global LoRA
        self.lora_global = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank_global, 
            alpha_global, 
            dropout
        )
        
        # Local LoRA
        self.lora_local = LoRALayer(
            linear.in_features, 
            linear.out_features, 
            rank_local, 
            alpha_local, 
            dropout
        )

    def forward(self, x):
        if self.lora_global.rank > 0 and self.lora_local.rank > 0:
            # Apply both global and local LoRA to the input
            return self.linear(x) + 0.5 * self.lora_global(x) + 0.5 * self.lora_local(x)
        elif self.lora_global.rank > 0:
            # Apply only global LoRA
            return self.linear(x) + self.lora_global(x)
        elif self.lora_local.rank > 0:
            # Apply only local LoRA
            return self.linear(x) + self.lora_local(x)
        else:
            # No LoRA applied
            return self.linear(x)
        

class CLIPModelWithLoRACombined(torch.nn.Module):
    def __init__(self, model_checkpoint, home_dir, lora_params_global, lora_params_local, momentum_global=0.1, momentum_local=0.5):
        """
        Initialize the CLIP model with combined LoRA adapters for both global and local.
        """
        super().__init__()
        self.model_checkpoint = model_checkpoint
        self.home_dir = home_dir
        self.lora_params_global = lora_params_global
        self.lora_params_local = lora_params_local
        self.momentum_global = momentum_global
        self.momentum_local = momentum_local
        home_dir = Path(home_dir)
        cache_dir = home_dir / "models"

        # Initialize two separate CLIP models
        self.model_global = CLIPModel.from_pretrained(self.model_checkpoint, cache_dir=cache_dir)
        self.model_combined = CLIPModel.from_pretrained(self.model_checkpoint, cache_dir=cache_dir)

        # Freeze all layers of both CLIP models
        for param in self.model_global.parameters():
            param.requires_grad = False
        for param in self.model_combined.parameters():
            param.requires_grad = False

        self.lora_layers_global = {}
        self.lora_layers_global_copy = {}  # New: copy of global LoRA layers
        self.lora_layers_local = {}

        # Apply LoRA layers to global model
        self._apply_lora_global()

        # Copy global LoRA layers to the copy dictionary
        self._copy_lora_global()

        # Apply both global copy and local LoRA to combined model
        self._apply_lora_combined()

        # Freeze all parameters of lora_layers_global_copy
        self._freeze_lora_global_copy()
        
        print(f'self.lora_layers_global: {self.lora_layers_global}')
        print(f'self.lora_layers_global_copy: {self.lora_layers_global_copy}')
        print(f'self.lora_layers_local: {self.lora_layers_local}')
        
        
        print(f'self.model_combined: {self.model_combined}')
        

    def _apply_lora_global(self):
        """
        Apply LoRA modifications to the CLIP model. This method initializes
        LoRA layers and replaces the corresponding layers in the CLIP model.
        """
        
        assign_lora = partial(
            LinearWithLoRA,
            rank=self.lora_params_global['rank'],
            alpha=self.lora_params_global['alpha'],
            dropout=self.lora_params_global['dropout']
        )
        
        def assign_and_store_lora(layer, attr, layer_name):
            try:
                lora_layer = assign_lora(getattr(layer, attr))
                setattr(layer, attr, lora_layer)
                
                # Store lora_layer parameters in the lora_layers dictionary
                for param_name, param in lora_layer.lora.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers_global[full_param_name] = param
                
            except AttributeError as e:
                print(f"Error: {e}")
                print(f"Available attributes in the layer: {dir(layer)}")
                    
        # Apply LoRA modifications to the text model's encoder layers
        for i, layer in enumerate(self.model_global.text_model.encoder.layers):
            if self.lora_params_global.get('lora_key_text', False):
                assign_and_store_lora(layer.self_attn, 'k_proj', f"text_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params_global.get('lora_query_text', False):
                assign_and_store_lora(layer.self_attn, 'q_proj', f"text_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params_global.get('lora_value_text', False):
                assign_and_store_lora(layer.self_attn, 'v_proj', f"text_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params_global.get('lora_outproj_text', False):
                assign_and_store_lora(layer.self_attn, 'out_proj', f"text_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params_global.get('lora_mlp_text', False):
                assign_and_store_lora(layer.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora(layer.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params_global.get('lora_head_text', False):
            self.model_global.text_projection = assign_lora(self.model_global.text_projection)
            for param_name, param in self.model_global.text_projection.lora.named_parameters():
                full_param_name = f"text_projection.{param_name}"
                self.lora_layers_global[full_param_name] = param

        # Apply LoRA modifications to the vision model's encoder layers
        for i, layer in enumerate(self.model_global.vision_model.encoder.layers):
            if self.lora_params_global.get('lora_key_vision', False):
                assign_and_store_lora(layer.self_attn, 'k_proj', f"vision_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params_global.get('lora_query_vision', False):
                assign_and_store_lora(layer.self_attn, 'q_proj', f"vision_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params_global.get('lora_value_vision', False):
                assign_and_store_lora(layer.self_attn, 'v_proj', f"vision_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params_global.get('lora_outproj_vision', False):
                assign_and_store_lora(layer.self_attn, 'out_proj', f"vision_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params_global.get('lora_mlp_vision', False):
                assign_and_store_lora(layer.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora(layer.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params_global.get('lora_head_vision', False):
            self.model_global.visual_projection = assign_lora(self.model_global.visual_projection)
            for param_name, param in self.model_global.visual_projection.lora.named_parameters():
                full_param_name = f"visual_projection.{param_name}"
                self.lora_layers_global[full_param_name] = param

    def _copy_lora_global(self):
        """
        Copy the global LoRA layers to a new dictionary `lora_layers_global_copy`.
        """
        for param_name, param in self.lora_layers_global.items():
            self.lora_layers_global_copy[param_name] = param.clone().detach().requires_grad_(True)

    def _apply_lora_combined(self):
        """
        Apply both global copy and local LoRA to the model_combined.
        """
        assign_lora_combined = partial(
            LinearWithLoRACombined,  # This will apply the global copy and local LoRA
            rank_global=self.lora_params_global['rank'],
            alpha_global=self.lora_params_global['alpha'],
            rank_local=self.lora_params_local['rank'],
            alpha_local=self.lora_params_local['alpha'],
            dropout=self.lora_params_global['dropout']
        )
        

        def assign_and_store_lora_combined(layer, attr, layer_name):
            try:
                lora_layer_combined = assign_lora_combined(getattr(layer, attr))
                setattr(layer, attr, lora_layer_combined)

                # Store global copy LoRA parameters
                for param_name, param in lora_layer_combined.lora_global.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers_global_copy[full_param_name] = param

                # Store local LoRA parameters
                for param_name, param in lora_layer_combined.lora_local.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers_local[full_param_name] = param

            except AttributeError as e:
                print(f"Error: {e}")
                print(f"Available attributes in the layer: {dir(layer)}")

        # Apply combined LoRA (both global copy and local) to the model's text and vision encoders
        for i, layer in enumerate(self.model_combined.text_model.encoder.layers):
            if self.lora_params_global.get('lora_key_text', False):
                assign_and_store_lora_combined(layer.self_attn, 'k_proj', f"text_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params_global.get('lora_query_text', False):
                assign_and_store_lora_combined(layer.self_attn, 'q_proj', f"text_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params_global.get('lora_value_text', False):
                assign_and_store_lora_combined(layer.self_attn, 'v_proj', f"text_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params_global.get('lora_outproj_text', False):
                assign_and_store_lora_combined(layer.self_attn, 'out_proj', f"text_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params_global.get('lora_mlp_text', False):
                assign_and_store_lora_combined(layer.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora_combined(layer.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params_global.get('lora_head_text', False):
            self.model_combined.text_projection = assign_lora_combined(self.model_combined.text_projection)
            for param_name, param in self.model_combined.text_projection.lora_global.named_parameters():
                self.lora_layers_global_copy[f'text_projection.{param_name}'] = param
            for param_name, param in self.model_combined.text_projection.lora_local.named_parameters():
                self.lora_layers_local[f'text_projection.{param_name}'] = param

        # Apply combined LoRA to the vision encoder
        for i, layer in enumerate(self.model_combined.vision_model.encoder.layers):
            if self.lora_params_global.get('lora_key_vision', False):
                assign_and_store_lora_combined(layer.self_attn, 'k_proj', f"vision_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params_global.get('lora_query_vision', False):
                assign_and_store_lora_combined(layer.self_attn, 'q_proj', f"vision_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params_global.get('lora_value_vision', False):
                assign_and_store_lora_combined(layer.self_attn, 'v_proj', f"vision_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params_global.get('lora_outproj_vision', False):
                assign_and_store_lora_combined(layer.self_attn, 'out_proj', f"vision_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params_global.get('lora_mlp_vision', False):
                assign_and_store_lora_combined(layer.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora_combined(layer.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params_global.get('lora_head_vision', False):
            self.model_combined.visual_projection = assign_lora_combined(self.model_combined.visual_projection)
            for param_name, param in self.model_combined.visual_projection.lora_global.named_parameters():
                self.lora_layers_global_copy[f'visual_projection.{param_name}'] = param
            for param_name, param in self.model_combined.visual_projection.lora_local.named_parameters():
                self.lora_layers_local[f'visual_projection.{param_name}'] = param

    def _freeze_lora_global_copy(self):
        """
        Freeze all parameters of the `lora_layers_global_copy` dictionary.
        """
        for param_name, param in self.lora_layers_global_copy.items():
            param.requires_grad = False

    def set_lora_dict_global(self, dictionary):
        """
        Set the parameters of both the lora_layers_global and lora_layers_global_copy from a dictionary.

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
        # Update lora_layers_global
        for key, param in self.lora_layers_global.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            param.data.copy_(dictionary[key].data)

        # Update lora_layers_global_copy
        for key, param in self.lora_layers_global_copy.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary for lora_layers_global_copy.")
            param.data.copy_(dictionary[key].data)
            
    def set_lora_dict_global_with_momentum(self, dictionary):
        """
        Set the parameters of both lora_layers_global and lora_layers_global_copy from a dictionary with momentum.

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
        # Update lora_layers_global
        for key, param in self.lora_layers_global.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            # param.data.copy_(dictionary[key].data)
            param.data.copy_(self.momentum_global * param.data + (1 - self.momentum_global) * dictionary[key].data)

        # Update lora_layers_global_copy
        for key, param in self.lora_layers_global_copy.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary for lora_layers_global_copy.")
                
            # param.data.copy_(dictionary[key].data)
            param.data.copy_(self.momentum_global * param.data + (1 - self.momentum_global) * dictionary[key].data)
            
    def set_lora_dict_local_with_momentum(self, dictionary):
        """
        Set the parameters of the lora_layers_local from a dictionary with momentum.

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
        # Update lora_layers_local
        for key, param in self.lora_layers_local.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            # param.data.copy_(dictionary[key].data)
            param.data.copy_(self.momentum_local * param.data + (1 - self.momentum_local) * dictionary[key].data)
    

    
    
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
    
