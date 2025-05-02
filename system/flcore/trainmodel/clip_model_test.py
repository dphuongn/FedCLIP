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


class CLIPModelWithLoRA(torch.nn.Module):
    def __init__(self, model_checkpoint, home_dir, lora_params, layer_keys =[]):
        """
        Initialize the CLIP model with LoRA layers.
        
        Args:
            model_checkpoint (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
            lora_params (dict): Parameters for configuring the LoRA layers.
        """
        super().__init__()
        self.model_checkpoint = model_checkpoint
        self.home_dir = home_dir
        self.lora_params = lora_params
        home_dir = Path(home_dir)
        cache_dir = home_dir / "models"
        
        self.model = CLIPModel.from_pretrained(self.model_checkpoint, cache_dir=cache_dir)
        # self.processor = CLIPProcessor.from_pretrained(self.model_checkpoint, cache_dir=f"{self.home_dir}/models")
        
        # Freeze all layers of CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.lora_layers = {}
        self.wa_layers = {}
        self.wb_layers = {}
        self._apply_lora()
        
        self.layer_keys = layer_keys

        self.lower_keys, self.higher_keys = self.filter_keys()

        # self.lower_keys = [k for k in self.lora_layers.keys() if not any(layer_key in k for layer_key in self.layer_keys)]
        # self.higher_keys = [k for k in self.lora_layers.keys() if any(layer_key in k for layer_key in self.layer_keys)]

        self.base = {k: self.lora_layers[k] for k in self.lower_keys}
        self.head = {k: self.lora_layers[k] for k in self.higher_keys}
        
        # print(f'self.lora_layers: {self.lora_layers}')
        
        # print(f'whole clip model: {self.model}')

    def filter_keys(self):
        layer_names = []
        layer_indices = []

        # Separate layer names and layer indices
        for key in self.layer_keys:
            if key.isdigit():
                layer_indices.append(int(key))
            elif '-' in key:
                start, end = map(int, key.split('-'))
                layer_indices.extend(range(start, end + 1))
            else:
                layer_names.append(key)

        # Convert indices to strings for matching
        layer_indices = [str(i) for i in layer_indices]

        # Filter lower_keys to include only the keys that do not contain any of the specified layer keys or indices
        lower_keys = [
            k for k in self.lora_layers.keys()
            if not any(layer_name in k for layer_name in layer_names)
            and not any(re.search(rf'\.{idx}\.', k) for idx in layer_indices)
        ]

        # Filter higher_keys to include only the keys that contain any of the specified layer keys or indices
        higher_keys = [
            k for k in self.lora_layers.keys()
            if any(layer_name in k for layer_name in layer_names)
            or any(re.search(rf'\.{idx}\.', k) for idx in layer_indices)
        ]

        return lower_keys, higher_keys
        
    def count_lora_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def count_lora_parameters_head(self):
        count = 0
        for key, param in self.lora_layers.items():
            if f"projection" in key:
                count += param.numel()
        return count
    
    def count_lora_parameters_layer(self, layer_index):
        count = 0
        for key, param in self.lora_layers.items():
            if f"layers.{layer_index}." in key:
                count += param.numel()
        return count
    
    def calculate_lora_size(self):
        param_size = 0
        for param in self.lora_layers.values():
            if param.requires_grad:
                param_size += param.nelement() * param.element_size()

        size_all_mb = param_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb
    
    def calculate_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024 ** 2  # Convert to megabytes
        return size_all_mb
    
    def calculate_lora_size_head(self):
        param_size = 0
        for key, param in self.lora_layers.items():
            if f"projection" in key:
                param_size += param.nelement() * param.element_size()

        size_all_mb = param_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb
    
    def calculate_lora_size_layer(self, layer_index):
        param_size = 0
        for key, param in self.lora_layers.items():
            if f"layers.{layer_index}." in key:
                param_size += param.nelement() * param.element_size()

        size_all_mb = param_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb
    
    def calculate_state_dict_size(self, state_dict):
        
        total_size = 0
    
        for name, layer in state_dict.items():
            if isinstance(layer, nn.Module):
                for param in layer.parameters():
                    total_size += param.nelement() * param.element_size()
            elif isinstance(layer, nn.Parameter):
                total_size += layer.nelement() * layer.element_size()

        size_all_mb = total_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb

        
    def _apply_lora(self):
        """
        Apply LoRA modifications to the CLIP model. This method initializes
        LoRA layers and replaces the corresponding layers in the CLIP model.
        """
        
        assign_lora = partial(
            LinearWithLoRA,
            rank=self.lora_params['rank'],
            alpha=self.lora_params['alpha'],
            dropout=self.lora_params['dropout']
        )
        
        def assign_and_store_lora(layer, attr, layer_name):
            try:
                lora_layer = assign_lora(getattr(layer, attr))
                setattr(layer, attr, lora_layer)
                
                # Store lora_layer parameters in the lora_layers dictionary
                for param_name, param in lora_layer.lora.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    self.lora_layers[full_param_name] = param
                
                    if 'W_a' in param_name:
                        self.wa_layers[full_param_name] = param
                    elif 'W_b' in param_name:
                        self.wb_layers[full_param_name] = param
                
                # if isinstance(lora_layer, LinearWithLoRA):
                #     self.lora_layers[layer_name] = lora_layer.lora
                # else:
                #     self.lora_layers[layer_name] = lora_layer
                
            except AttributeError as e:
                print(f"Error: {e}")
                print(f"Available attributes in the layer: {dir(layer)}")
                # if 'self_attn' in dir(layer):
                #     print(f"Available attributes in self_attn: {dir(layer.self_attn)}")
                # else:
                #     print(f"Available attributes in the layer: {dir(layer)}")
                    
        # Apply LoRA modifications to the text model's encoder layers
        for i, layer in enumerate(self.model.text_model.encoder.layers):
            if self.lora_params.get('lora_key_text', False):
                assign_and_store_lora(layer.self_attn, 'k_proj', f"text_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params.get('lora_query_text', False):
                assign_and_store_lora(layer.self_attn, 'q_proj', f"text_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params.get('lora_value_text', False):
                assign_and_store_lora(layer.self_attn, 'v_proj', f"text_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params.get('lora_outproj_text', False):
                assign_and_store_lora(layer.self_attn, 'out_proj', f"text_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params.get('lora_mlp_text', False):
                assign_and_store_lora(layer.mlp, 'fc1', f"text_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora(layer.mlp, 'fc2', f"text_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params.get('lora_head_text', False):
            self.model.text_projection = assign_lora(self.model.text_projection)
            for param_name, param in self.model.text_projection.lora.named_parameters():
                full_param_name = f"text_projection.{param_name}"
                self.lora_layers[full_param_name] = param
                if 'W_a' in param_name:
                    self.wa_layers[full_param_name] = param
                elif 'W_b' in param_name:
                    self.wb_layers[full_param_name] = param

        # Apply LoRA modifications to the vision model's encoder layers
        for i, layer in enumerate(self.model.vision_model.encoder.layers):
            if self.lora_params.get('lora_key_vision', False):
                assign_and_store_lora(layer.self_attn, 'k_proj', f"vision_model.encoder.layers.{i}.self_attn.k_proj")
            if self.lora_params.get('lora_query_vision', False):
                assign_and_store_lora(layer.self_attn, 'q_proj', f"vision_model.encoder.layers.{i}.self_attn.q_proj")
            if self.lora_params.get('lora_value_vision', False):
                assign_and_store_lora(layer.self_attn, 'v_proj', f"vision_model.encoder.layers.{i}.self_attn.v_proj")
            if self.lora_params.get('lora_outproj_vision', False):
                assign_and_store_lora(layer.self_attn, 'out_proj', f"vision_model.encoder.layers.{i}.self_attn.out_proj")
            if self.lora_params.get('lora_mlp_vision', False):
                assign_and_store_lora(layer.mlp, 'fc1', f"vision_model.encoder.layers.{i}.mlp.fc1")
                assign_and_store_lora(layer.mlp, 'fc2', f"vision_model.encoder.layers.{i}.mlp.fc2")

        if self.lora_params.get('lora_head_vision', False):
            self.model.visual_projection = assign_lora(self.model.visual_projection)
            for param_name, param in self.model.visual_projection.lora.named_parameters():
                full_param_name = f"visual_projection.{param_name}"
                self.lora_layers[full_param_name] = param
                if 'W_a' in param_name:
                    self.wa_layers[full_param_name] = param
                elif 'W_b' in param_name:
                    self.wb_layers[full_param_name] = param
        
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
        for key, param in self.lora_layers.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            param.data.copy_(dictionary[key].data)
            
    def set_base_dict(self, dictionary):
        """
        Set the parameters of the lower LoRA layers from a dictionary.

        Args:
            dictionary (dict): A dictionary containing parameters for the lower LoRA layers.
        """
        for key in self.lower_keys:
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")
            if key not in self.lora_layers:
                raise KeyError(f"Parameter key {key} not found in self.lora_layers.")
        
            self.lora_layers[key].data.copy_(dictionary[key].data)

    def set_head_dict(self, dictionary):
        """
        Set the parameters of the higher LoRA layers from a dictionary.

        Args:
            dictionary (dict): A dictionary containing parameters for the higher LoRA layers.
        """
        for key in self.higher_keys:
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")
            if key not in self.lora_layers:
                raise KeyError(f"Parameter key {key} not found in self.lora_layers.")
        
            self.lora_layers[key].data.copy_(dictionary[key].data)
            
    def freeze_lora_layers(self, keys_to_freeze: list[str]) -> None:
        """
        Freeze certain layers of lora_layers according to specified keys.

        Args:
            keys_to_freeze (List[str]): List of keys corresponding to the layers to be frozen.
        """
        for key in keys_to_freeze:
            if key in self.lora_layers:
                self.lora_layers[key].requires_grad = False
            else:
                raise KeyError(f"Parameter key {key} not found in lora_layers.")
                
    def unfreeze_lora_layers(self, keys_to_unfreeze: list[str]) -> None:
        """
        Unfreeze certain layers of lora_layers according to specified keys.

        Args:
            keys_to_unfreeze (List[str]): List of keys corresponding to the layers to be unfrozen.
        """
        for key in keys_to_unfreeze:
            if key in self.lora_layers:
                self.lora_layers[key].requires_grad = True
            else:
                raise KeyError(f"Parameter key {key} not found in lora_layers.")

    def set_lora_dict_with_momentum(self, global_lora_params, momentum):
        for key, param in self.lora_layers.items():
            if key not in global_lora_params:
                raise KeyError(f"Parameter key {key} not found in the provided global parameters.")

            param.data.copy_(momentum * param.data + (1 - momentum) * global_lora_params[key].data)

    def set_wa_dict(self, dictionary):
        """
        Set the parameters of the W_a layers from a dictionary.

        Args:
            dictionary (dict): A dictionary containing parameters for the W_a layers.

        Others:
            dictionary structure: 
            {'text_model.encoder.layers.0.self_attn.k_proj.W_a': tensor(...),
            'text_model.encoder.layers.1.self_attn.q_proj.W_a': tensor(...),
            'text_model.encoder.layers.2.self_attn.v_proj.W_a': tensor(...),
            'text_model.encoder.layers.2.self_attn.out_proj.W_a': tensor(...),
            'text_model.encoder.layers.4.mlp.fc1.W_a': tensor(...),
            'text_projection.W_a': tensor(...),
            ...}
        """
        for key, param in self.wa_layers.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            param.data.copy_(dictionary[key].data)

    def set_wb_dict(self, dictionary):
        """
        Set the parameters of the W_b layers from a dictionary.

        Args:
            dictionary (dict): A dictionary containing parameters for the W_b layers.

        Others:
            dictionary structure: 
            {'text_model.encoder.layers.0.self_attn.k_proj.W_b': tensor(...),
            'text_model.encoder.layers.1.self_attn.q_proj.W_b': tensor(...),
            'text_model.encoder.layers.2.self_attn.v_proj.W_b': tensor(...),
            'text_model.encoder.layers.3.self_attn.out_proj.W_b': tensor(...),
            'text_model.encoder.layers.4.mlp.fc1.W_b': tensor(...),
            'text_projection.W_b': tensor(...),
            ...}
        """
        for key, param in self.wb_layers.items():
            if key not in dictionary:
                raise KeyError(f"Parameter key {key} not found in the provided dictionary.")

            param.data.copy_(dictionary[key].data)

    def print_lora_dict_shapes(self, dictionary):
        """
        Print the shapes of tensors stored in a dictionary.

        This function iterates over each key-value pair in the dictionary.
        It assumes that each value is a tensor and prints the shape of each tensor
        along with its corresponding key.

        Args:
            dictionary (dict): A dictionary where each value is expected to be a tensor.
        """
        for key, param in dictionary.items():
            print(f"Shape of '{key}': {param.shape}")
            # print(f"Details of '{key}': {param}")
            print(f'name of {key}: {key}')
                
            
    def compare_lora_dicts(self, dict1, dict2, tolerance=1e-6):
        """
        Compare two dictionaries containing LoRA parameters.

        Args:
            dict1 (dict): The first dictionary of LoRA parameters.
            dict2 (dict): The second dictionary of LoRA parameters.
            tolerance (float): Tolerance level for comparing floating point values.

        Returns:
            bool: True if the dictionaries are the same within the given tolerance, False otherwise.
        """

        if dict1.keys() != dict2.keys():
            return False

        for key in dict1:
            param1 = dict1[key]
            param2 = dict2[key]

            if not torch.allclose(param1, param2, atol=tolerance):
                return False

        return True  

    def count_trainable_params(self) -> int:
        """
        Returns the total number of trainable parameters (i.e. requires_grad=True)
        in the entire CLIPModelWithDualLoRA.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_trainable_params(self):
        """
        Prints a nicely formatted count of trainable parameters.
        """
        total = self.count_trainable_params()
        print(f"[LoRA CLIP] Total trainable parameters: {total:,}")


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

        # 'lora_query_text': True,
        # 'lora_key_text': True,
        # 'lora_value_text': True,
        # 'lora_outproj_text': True,
        # 'lora_mlp_text': True,
        # 'lora_head_text': True,
        

        'lora_query_vision': True,
        'lora_key_vision': True,
        'lora_value_vision': True,
        'lora_outproj_vision': True,
        'lora_mlp_vision': True,
        'lora_head_vision': True,
    }


    # test CLIPModelWithLoRA
    
    CLIPModelWithLoRA_object = CLIPModelWithLoRA(model_checkpoint=model_checkpoint, home_dir=HOME, lora_params=lora_params).to(device)
    
    # model = CLIPModelWithLoRA_object.model
    
    # lora_layers = CLIPModelWithLoRA_object.lora_layers

    CLIPModelWithLoRA_object.print_trainable_params()