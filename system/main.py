import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

# from flcore.servers.serverfft import FedFft
from flcore.servers.serverlora import FLora
from flcore.servers.servermeta import FMeta
from flcore.servers.serverloraselectmost import FLoraSelectMost
from flcore.servers.serverloraselectmostwoh import FLoraSelectMostWoH
from flcore.servers.serverloraselectleast import FLoraSelectLeast
from flcore.servers.serverloraselectleastwoh import FLoraSelectLeastWoH
from flcore.servers.serverloraselecthead import FLoraSelectHead
from flcore.servers.serverloraselectnonhead import FLoraSelectNonHead
from flcore.servers.serverloraselectrandom import FLoraSelectRandom
# from flcore.servers.serverloradp import FLoradp
from flcore.servers.serverloralocal import FLoraLocal
from flcore.servers.serversoralocal import FSoraLocal
# from flcore.servers.serversoradual import FSoraDual
# from flcore.servers.serversoradualhetero import FSoraDualHetero
# from flcore.servers.serversoradualheterotrue import FSoraDualHeteroTrue
# from flcore.servers.serversoradualheterotruemomen import FSoraDualHeteroTrueMomen
# from flcore.servers.serversoradualheterotruemu import FSoraDualHeteroTrueMu
# from flcore.servers.serveraa import FedAa
# from flcore.servers.serverloradat import FLoraDAT
# from flcore.servers.serverloradattrue import FLoraDATTrue
# from flcore.servers.serverloradattruemomen import FLoraDATTrueMomen
# from flcore.servers.serverloradatglobal import FLoraDATGlobal
# from flcore.servers.server2prune import FLora2Prune
# from flcore.servers.server2pruneglobal import FLora2PruneGlobal
# from flcore.servers.server2prunereverse import FLora2PruneReverse
# from flcore.servers.serverprunelora import FLoraPruneLora
# from flcore.servers.serverpruneblock import FLoraPruneBlock
# from flcore.servers.serverpruneblocklocal import FLoraPruneBlockLocal
# from flcore.servers.serverprunemodulelocal import FLoraPruneModuleLocal
# from flcore.servers.serverlc import FedLc
# from flcore.servers.servervmlc import FedVmLc
# from flcore.servers.serverprompt import FedPrompt
# from flcore.servers.serverca import FedCa



from flcore.trainmodel.clip_model import *
from flcore.trainmodel.clip_model_simple import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

# from dataset_config import get_dataset_name, get_class_names
from utils.config import get_dataset_name, get_class_names, set_random_seed

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

import wandb  # Import wandb


def setup_wandb(args):
    # Initialize wandb for the project
    wandb.init(
        project="your_project_name",  # Replace with your project name
        config={
            "local_learning_rate": args.local_learning_rate,
            "learning_rate_decay": args.learning_rate_decay,
            "learning_rate_decay_gamma": args.learning_rate_decay_gamma,
            # "local_epochs": args.local_epochs,
            # Add more hyperparameters you want to sweep
        }
    )
    return wandb.config

def run(args):
    
    # Initialize WandB
    # wandb_config = setup_wandb(args)

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(0, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        
        if args.model == "vit-b-32":
            args.model_checkpoint = "openai/clip-vit-base-patch32"
        elif args.model == "vit-b-16":
            args.model_checkpoint = "openai/clip-vit-base-patch16"
        elif args.model == "vit-l-14":
            args.model_checkpoint = "openai/clip-vit-large-patch14"
        elif args.model == "vit-l-14-336":
            args.model_checkpoint = "openai/clip-vit-large-patch14-336"
        else:
            raise NotImplementedError
            
        args.processor = get_processor(args.model_checkpoint, args.home_dir)
            

        if args.algorithm == "flora":
            server = FLora(args, i)
        
        elif args.algorithm == "fmeta":
            server = FMeta(args, i)

        elif args.algorithm == "floraselectmost":
            server = FLoraSelectMost(args, i)

        elif args.algorithm == "floraselectmostwoh":
            server = FLoraSelectMostWoH(args, i)

        elif args.algorithm == "floraselectleast":
            server = FLoraSelectLeast(args, i)

        elif args.algorithm == "floraselectleastwoh":
            server = FLoraSelectLeastWoH(args, i)

        elif args.algorithm == "floraselecthead":
            server = FLoraSelectHead(args, i)

        elif args.algorithm == "floraselectnonhead":
            server = FLoraSelectNonHead(args, i)

        elif args.algorithm == "floraselectrandom":
            server = FLoraSelectRandom(args, i)

        elif args.algorithm == "floradp":
            server = FLoradp(args, i)

        elif args.algorithm == "floralocal":
            server = FLoraLocal(args, i)
        
        elif args.algorithm == "fsoralocal":
            server = FSoraLocal(args, i)

        elif args.algorithm == "fsoradual":
            server = FSoraDual(args, i)

        elif args.algorithm == "fsoradualhetero":
            server = FSoraDualHetero(args, i)

        elif args.algorithm == "fsoradualheterotrue":
            server = FSoraDualHeteroTrue(args, i)
        
        elif args.algorithm == "fsoradualheterotruemomen":
            server = FSoraDualHeteroTrueMomen(args, i)

        elif args.algorithm == "fsoradualheterotruemu":
            server = FSoraDualHeteroTrueMu(args, i)
            
        elif args.algorithm == "fedaa":
            server = FedAa(args, i)

        elif args.algorithm == "floradat":
            server = FLoraDAT(args, i)

        elif args.algorithm == "floradattrue":
            server = FLoraDATTrue(args, i)

        elif args.algorithm == "floradattruemomen":
            server = FLoraDATTrueMomen(args, i)

        elif args.algorithm == "floradatglobal":
            server = FLoraDATGlobal(args, i)

        elif args.algorithm == "flora2prune":
            server = FLora2Prune(args, i)

        elif args.algorithm == "flora2prunereverse":
            server = FLora2PruneReverse(args, i)

        elif args.algorithm == "flora2pruneglobal":
            server = FLora2PruneGlobal(args, i)

        elif args.algorithm == "floraprunelora":
            server = FLoraPruneLora(args, i)

        elif args.algorithm == "florapruneblock":
            server = FLoraPruneBlock(args, i)

        elif args.algorithm == "florapruneblocklocal":
            server = FLoraPruneBlockLocal(args, i)
            
        elif args.algorithm == "floraprunemodulelocal":
            server = FLoraPruneModuleLocal(args, i)
        
        elif args.algorithm == "floramu":
            server = FLoraMu(args, i, wandb_config)

        elif args.algorithm == "fedwa":
            server = FedWa(args, i, wandb_config)

        elif args.algorithm == "fedwb":
            server = FedWb(args, i, wandb_config)
            
        elif args.algorithm == "fdylora":
            server = FedDyLora(args, i, wandb_config)
            
        elif args.algorithm == "feddora":
            server = FedDora(args, i, wandb_config)
            
        elif args.algorithm == "fdydora":
            server = FedDyDora(args, i, wandb_config)

        elif args.algorithm == "floraprox":
            server = FLoraProx(args, i, wandb_config)

        elif args.algorithm == "floraditto":
            server = FLoraDitto(args, i, wandb_config)
            
        elif args.algorithm == "floraala":
            server = FLoraAla(args, i, wandb_config)
            
        elif args.algorithm == "floramomentum":
            server = FLoraMomentum(args, i, wandb_config)
            
        elif args.algorithm == "floraper":
            server = FLoraPer(args, i, wandb_config)

        elif args.algorithm == "florababu":
            server = FLoraBabu(args, i, wandb_config)
        
        elif args.algorithm == "florapcl":
            server = FLoraPcl(args, i, wandb_config)

        elif args.algorithm == "floradbe":
            server = FLoraDbe(args, i, wandb_config)
            
        elif args.algorithm == "floramoon":
            server = FLoraMoon(args, i, wandb_config)
            
        elif args.algorithm == "floraprune":
            server = FLoraPrune(args, i, wandb_config)

        elif args.algorithm == "florablockprune":
            server = FLoraBlockPrune(args, i, wandb_config)
            
        elif args.algorithm == "florablockprune2":
            server = FLoraBlockPrune2(args, i, wandb_config)
            
        elif args.algorithm == "floraproto":
            server = FLoraProto(args, i, wandb_config)
            
        elif args.algorithm == "floraflyp":
            server = FLoraFlyp(args, i, wandb_config)

        elif args.algorithm == "fedfft":
            server = FedFft(args, i, wandb_config)
            
        elif args.algorithm == "fedlc":
            server = FedLc(args, i, wandb_config)
        
        elif args.algorithm == "fedvmlc":
            server = FedVmLc(args, i, wandb_config)
            
        elif args.algorithm == "fedprompt":
            server = FedPrompt(args, i, wandb_config)
            
        elif args.algorithm == "fedca":
            server = FedCa(args, i, wandb_config)
        
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, times=args.times, delete=args.delete)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print('__TORCH Version:', torch.__version__)
        print('__CUDNN Version:', torch.backends.cudnn.version())
        print("__CUDA version: ", torch.version.cuda)
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory/1e9)
        
        
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-m', "--model", type=str, default="vit-b-32")
    parser.add_argument('-lbs', "--batch_size_train", type=int, default=512)
    parser.add_argument('-bs_test', "--batch_size_test", type=int, default=512)

    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.00005,
                        help="Local learning rate")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('-wd', "--weight_decay", type=float, default=0)
    parser.add_argument('-ld', "--learning_rate_decay", action='store_true', help="apply learning rate decay")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    # meta
    parser.add_argument('-inner_steps', "--inner_steps", type=int, default=3)
    parser.add_argument('-meta_inner_lr', "--meta_inner_lr", type=float, default=2e-2)
    parser.add_argument('-meta_outer_lr', "--meta_outer_lr", type=float, default=5e-4)
    parser.add_argument('-gumbel_temp', "--gumbel_temp", type=float, default=0.5)
    parser.add_argument('-meta_support_fraction', "--meta_support_fraction", type=float, default=0.5)

    parser.add_argument('-kl_gamma', "--kl_gamma", type=float, default=1.0, help="weight on KL distillation loss")
    parser.add_argument('-consistency_lambda', "--consistency_lambda", type=float, default=0.1, help="weight on embedding‐consistency loss")
    parser.add_argument('-sparsity_lambda', "--sparsity_lambda", type=float, default=1e-4, help="L1 penalty on gates")
    parser.add_argument('-server_lr', "--server_lr", type=float, default=0.5, help="global meta‐step size (0.5=half‐way toward client mean)")
    parser.add_argument('-distill_temp', "--distill_temp", type=float, default=3.0, help="temperature for KL divergence")

    # FLoRADrop / FLoRADropReverse / FLoRASelectMost / FLoRASelectLeast
    parser.add_argument('-k_layers', "--k_layers", type=int, default=1)

    # FLoRAProx / MOON (for mu)
    parser.add_argument('-mu', "--mu", type=float, default=0.5)
    parser.add_argument('-mu_lr', "--mu_learning_rate", type=float, default=0.001,
                        help="Mu learning rate")

    # FLoRADitto
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)

    # FLoRAMomentum / FLoRADBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)

    # FLoRADBE
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    
    # FLoRAPrune
    parser.add_argument('-sr', "--sparsity_ratio", type=float, default=0.5)
    parser.add_argument('-ar', "--aggregation_ratio", type=float, default=1.5)

    #FLoRABlockPrune
    parser.add_argument('-pr', "--pruning_ratio", type=float, default=0.2)
    parser.add_argument('-pn', "--pruning_number", type=int, default=1)
    # parser.add_argument('-nbtp', "--num_blocks_to_prune", type=int, default=2)
    parser.add_argument('-train_df_bp', "--train_data_fraction_blockprune", type=float, default=1.0)
    parser.add_argument('-test_df_bp', "--test_data_fraction_blockprune", type=float, default=1.0)
    
    #FLoRAProto
    parser.add_argument('-lamda', "--lamda", type=float, default=1.0)
    
    # FLoRAAla / FLoRAPer / FedBabu
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=float, default=0.8)
    # parser.add_argument('-p', "--layer_idx", type=int, default=2,
    #                     help="More fine-grained than its original paper.")
    available_layer_keys = ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'mlp', 'text_projection']
    parser.add_argument('-p', "--layer_keys", type=str, nargs='+', default=['text_projection'],
                    help="Layer keys to decide which ones go to lower_keys. Available options: {', '.join(available_layer_keys)}.")

    # FedBabu
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=1)
    
    # MOON / FedPCL
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    
    
    # FLoRADAT (for mu)
    parser.add_argument('-mu_g', "--mu_global", type=float, default=0.5)
    parser.add_argument('-mu_lr_g', "--mu_learning_rate_global", type=float, default=0.001,
                        help="Global mu learning rate")
    parser.add_argument('-mu_l', "--mu_local", type=float, default=0.5)
    parser.add_argument('-mu_lr_l', "--mu_learning_rate_local", type=float, default=0.001,
                        help="Local mu learning rate")

    # FLoRADATTrueMomen
    parser.add_argument('-momen_g', "--momentum_global", type=float, default=0.1)
    parser.add_argument('-momen_l', "--momentum_local", type=float, default=0.5)
    
    # SoRA
    parser.add_argument('-s_l', "--sparse_lambda", type=float, default=1e-3,
                       help="clipping scale for gate param")
    parser.add_argument('-l_s', "--lambda_schedule", type=str, default=None,
                       help="scheduling of sparse_lambda, {linear, log_linear}")
    parser.add_argument('-m_l', "--max_lambda", type=float, default=10,
                       help="maximum value of sparse_lambda in scheduling")
    parser.add_argument('-l_n', "--lambda_num", type=float, default=10,
                       help="total number of lambdas in scheduling")

    # SoRADUALHETERO
    parser.add_argument('--lora_rank_local', type=int, default=4, help="LoRA rank local")
    parser.add_argument('-g_l', "--gamma_local", type=float, default=0.1, help="gamma for local SoRA -> global LoRA distillation")

    # DP-SGD
    parser.add_argument('-t_e', '--target_epsilon', type=float, default=2.0, help="target epsilon for privacy")

    
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", action='store_true',
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ftn', "--finetune_new_clients", action='store_true', help="Enable fine-tune new clients")
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=1)
    
    parser.add_argument('-train_d_f', "--train_data_fraction", type=float, default=1.0)
    parser.add_argument('-test_d_f', "--test_data_fraction", type=float, default=1.0)
    parser.add_argument('-sd', "--seed", type=int, default=0, help="Random seed")
    parser.add_argument('-pfl', "--personalized_fl", action='store_true', help="Enable Personalized Federated Learning")
    
    parser.add_argument('--no_normalize_weights', action='store_true', help="Do not normalize weights for aggregating parameters")

    # parser.add_argument('--home_dir', type=str, default=os.path.expanduser('~'))
    # parser.add_argument('--home_dir', type=str, default=str(Path.home()))

    # parser.add_argument('--home_dir', type=str, default="/scratch/bczq/")
    parser.add_argument('--home_dir', type=str, default="/work/LAS/jannesar-lab/dphuong")
    
    parser.add_argument('--delete', action='store_false', help="delete result.h5 file after reading")
    
    parser.add_argument('-norm_rank', '--normalized_rank',  action='store_true', help="Enable normalized LoRA ranks for fdylora")
    parser.add_argument('-norm_alpha', '--normalized_alpha',  action='store_true', help="Enable normalized LoRA alphas for fdylora")
    parser.add_argument('-rand_rank', '--random_rank', action='store_true', help="Enable LoRA random rank")
    parser.add_argument('-rand_alpha', '--random_alpha', action='store_true', help="Enable LoRA random alpha")
    parser.add_argument('-fi_rank', '--fixed_rank', action='store_true', help="Enable LoRA fixed rank")
    parser.add_argument('-fi_alpha', '--fixed_alpha', action='store_true', help="Enable LoRA fixed alpha")
    

    parser.add_argument('--lora_rank', type=int, default=2, help="LoRA rank")
    parser.add_argument('--lora_rank_min', type=int, default=0, help="LoRA rank min")
    parser.add_argument('--lora_rank_max', type=int, default=32, help="LoRA rank max")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--lora_alpha_min', type=int, default=0, help="LoRA alpha min")
    parser.add_argument('--lora_alpha_max', type=int, default=64, help="LoRA alpha max")
    parser.add_argument('--lora_dropout', type=float, default=0.00, help="LoRA dropout rate")
    
    parser.add_argument('--lora_key_text', action='store_true', help="LoRA apply to key text")
    parser.add_argument('--lora_query_text', action='store_true', help="LoRA apply to query text")
    parser.add_argument('--lora_value_text', action='store_true', help="LoRA apply to value text")
    parser.add_argument('--lora_outproj_text', action='store_true', help="LoRA apply to output projection (attention module) text")
    parser.add_argument('--lora_mlp_text', action='store_true', help="LoRA apply to MLP text")
    parser.add_argument('--lora_head_text', action='store_true', help="LoRA apply to head text")
    
    parser.add_argument('--lora_query_vision', action='store_true', help="LoRA apply to query vision")
    parser.add_argument('--lora_key_vision', action='store_true', help="LoRA apply to key vision")
    parser.add_argument('--lora_value_vision', action='store_true', help="LoRA apply to value vision")
    parser.add_argument('--lora_outproj_vision', action='store_true', help="LoRA apply to output projection (attention module) vision")
    parser.add_argument('--lora_mlp_vision', action='store_true', help="LoRA apply to MLP vision")
    parser.add_argument('--lora_head_vision', action='store_true', help="LoRA apply to head vision")
    
    
    parser.add_argument('-aa_b_r', '--aa_bottleneck_reduction', type=int, default=1, help="Attention Adapter bottleneck reduction")
    parser.add_argument('-aa_text', '--aa_text', action='store_true', help="Attention Adapter apply to text")
    parser.add_argument('-aa_vision', '--aa_vision', action='store_true', help="Attention Adapter apply to vision")
    
    
    parser.add_argument('--ca_bottleneck_reduction', type=int, default=4, help="Clip Adapter bottleneck reduction")
    parser.add_argument('--ca_ratio', type=float, default=0.2, help="Clip Adapter ratio")
    parser.add_argument('--ca_text', action='store_true', help="Clip Adapter apply to text")
    parser.add_argument('--ca_vision', action='store_true', help="Clip Adapter apply to vision")

    


    args = parser.parse_args()
    
    args.dataset = get_dataset_name(args.dataset)
    args.class_names = get_class_names(args.dataset)
    args.num_classes = len(args.class_names)    
    
    args.aa_params = {
        'aa_bottleneck_reduction': args.aa_bottleneck_reduction,
        'aa_text': args.aa_text,
        'aa_vision': args.aa_vision,
    }
    
    
    args.lora_params = {
        'rank': args.lora_rank,
        'rank_min': args.lora_rank_min,
        'rank_max': args.lora_rank_max,
        'alpha': args.lora_alpha,
        'alpha_min': args.lora_alpha_min,
        'alpha_max': args.lora_alpha_max,
        'dropout': args.lora_dropout,
        'lora_key_text': args.lora_key_text,
        'lora_query_text': args.lora_query_text,
        'lora_value_text': args.lora_value_text,
        'lora_outproj_text': args.lora_outproj_text,
        'lora_mlp_text': args.lora_mlp_text,
        'lora_head_text': args.lora_head_text,
        'lora_key_vision': args.lora_key_vision,
        'lora_query_vision': args.lora_query_vision,
        'lora_value_vision': args.lora_value_vision,
        'lora_outproj_vision': args.lora_outproj_vision,
        'lora_mlp_vision': args.lora_mlp_vision,
        'lora_head_vision': args.lora_head_vision,
    }

    args.lora_params_local = {
        'rank': args.lora_rank_local,
        'rank_min': args.lora_rank_min,
        'rank_max': args.lora_rank_max,
        'alpha': args.lora_alpha,
        'alpha_min': args.lora_alpha_min,
        'alpha_max': args.lora_alpha_max,
        'dropout': args.lora_dropout,
        'lora_key_text': args.lora_key_text,
        'lora_query_text': args.lora_query_text,
        'lora_value_text': args.lora_value_text,
        'lora_outproj_text': args.lora_outproj_text,
        'lora_mlp_text': args.lora_mlp_text,
        'lora_head_text': args.lora_head_text,
        'lora_key_vision': args.lora_key_vision,
        'lora_query_vision': args.lora_query_vision,
        'lora_value_vision': args.lora_value_vision,
        'lora_outproj_vision': args.lora_outproj_vision,
        'lora_mlp_vision': args.lora_mlp_vision,
        'lora_head_vision': args.lora_head_vision,
    }

    args.ca_params = {
        'bottleneck_reduction': args.ca_bottleneck_reduction,
        'ratio': args.ca_ratio,
        'aa_text': args.ca_text,
        'aa_vision': args.ca_vision,
    }
    
    
    set_random_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    
    print("Seed: {}".format(args.seed))
    print("Algorithm: {}".format(args.algorithm))
    print("Personalized Federated Learning: {}".format(args.personalized_fl))
    print("Delete result.h5 file after reading: {}".format(args.delete))
    print("Do NOT normalize weights for aggregating parameters: {}".format(args.no_normalize_weights))
    print("Train data fraction: {}".format(args.train_data_fraction))
    print("Test data fraction: {}".format(args.test_data_fraction))
    print("Running times: {}".format(args.times))
    print("Rounds gap for evaluation: {}".format(args.eval_gap))
    print("Dataset: {}".format(args.dataset))
    print("Class_names: {}".format(args.class_names))
    print("Number of classes: {}".format(args.num_classes))
    print("Checkpoint: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Rate for clients that train but drop out: {}".format(args.client_drop_rate))
    print("Number of new clients: {}".format(args.num_new_clients))
    print("Fine-tune new clients: {}".format(args.finetune_new_clients))
    
    
    # print("Normalized Rank: {}".format(args.normalized_rank))
    # print("Normalized Alpha: {}".format(args.normalized_alpha))
    
    print("lora_rank: {}".format(args.lora_rank))
    # print("lora_rank_min: {}".format(args.lora_rank_min))
    # print("lora_rank_max: {}".format(args.lora_rank_max))
    print("lora_alpha: {}".format(args.lora_alpha))
    # print("lora_alpha_min: {}".format(args.lora_alpha_min))
    # print("lora_alpha_max: {}".format(args.lora_alpha_max))
    print("lora_dropout: {}".format(args.lora_dropout))
    
    print("lora_key_text: {}".format(args.lora_key_text))
    print("lora_query_text: {}".format(args.lora_query_text))
    print("lora_value_text: {}".format(args.lora_value_text))
    print("lora_outproj_text: {}".format(args.lora_outproj_text))
    print("lora_mlp_text: {}".format(args.lora_mlp_text))
    print("lora_head_text: {}".format(args.lora_head_text))
    
    print("lora_key_vision: {}".format(args.lora_key_vision))
    print("lora_query_vision: {}".format(args.lora_query_vision))
    print("lora_value_vision: {}".format(args.lora_value_vision))
    print("lora_outproj_vision: {}".format(args.lora_outproj_vision))
    print("lora_mlp_vision: {}".format(args.lora_mlp_vision))
    print("lora_head_vision: {}".format(args.lora_head_vision))
    
    print("aa_bottleneck_reduction: {}".format(args.aa_bottleneck_reduction))
    print("aa_text: {}".format(args.aa_text))
    print("aa_vision: {}".format(args.aa_vision))
    
    print("ca_bottleneck_reduction: {}".format(args.ca_bottleneck_reduction))
    print("ca_text: {}".format(args.ca_text))
    print("ca_vision: {}".format(args.ca_vision))
    
    print(f"{'-'*5} For DAT {'-'*5}")
    print("mu_global: {}".format(args.mu_global))
    print("mu_learning_rate_global: {}".format(args.mu_learning_rate_global))
    print("mu_local: {}".format(args.mu_local))
    print("mu_learning_rate_local: {}".format(args.mu_learning_rate_local))
    print("-" * 20)

    print(f"{'-'*5} For Drop / DropReverse / SelectMost / SelectLeast {'-'*5}")
    print("k_layers: {}".format(args.k_layers))
    print("-" * 20)

    print(f"{'-'*5} For DATTrueMomen {'-'*5}")
    print("momentum_global: {}".format(args.momentum_global))
    print("momentum_local: {}".format(args.momentum_local))
    print("-" * 20)
    
    print(f"{'-'*5} For Prox {'-'*5}")
    print("mu: {}".format(args.mu))
    print("-" * 20)
    
    print(f"{'-'*5} For Ditto {'-'*5}")
    print("plocal_epochs: {}".format(args.plocal_epochs))
    print("-" * 20)
    
    print(f"{'-'*5} For Momentum {'-'*5}")
    print("momentum: {}".format(args.momentum))
    print("-" * 20)

    print(f"{'-'*5} For DBE {'-'*5}")
    print("kl_weight: {}".format(args.kl_weight))
    print("-" * 20)
    
    print(f"{'-'*5} For Prune {'-'*5}")
    print("sparsity ratio: {}".format(args.sparsity_ratio))
    print("aggregation ratio: {}".format(args.aggregation_ratio))
    print("pruning ratio: {}".format(args.pruning_ratio))
    print("pruning number: {}".format(args.pruning_number))
    # print("number of pruning layers: {}".format(args.num_blocks_to_prune))
    print("train data fraction blockprune: {}".format(args.train_data_fraction_blockprune))
    print("test data fraction blockprune: {}".format(args.test_data_fraction_blockprune))
    print("-" * 20)
    
    print(f"{'-'*5} For Proto {'-'*5}")
    print("lamda: {}".format(args.lamda))
    print("-" * 20)
    
    
    print(f"{'-'*5} For ALA {'-'*5}")
    print("eta: {}".format(args.eta))
    print("rand_percent: {}".format(args.rand_percent))
    print("layer_keys: {}".format(args.layer_keys))
    print("-" * 20)

    print(f"{'-'*5} For SoRA {'-'*5}")
    print("sparse_lambda: {}".format(args.sparse_lambda))
    print("lambda_schedule: {}".format(args.lambda_schedule))
    print("max_lambda: {}".format(args.max_lambda))
    print("lambda_num: {}".format(args.lambda_num))
    print("-" * 20)

    print(f"{'-'*5} For SoRADUALHETERO {'-'*5}")
    print("lora_rank_local: {}".format(args.lora_rank_local))
    print("gamma_local: {}".format(args.gamma_local))
    print("-" * 20)

    print(f"{'-'*5} For DP-SGD {'-'*5}")
    print("target_epsilon: {}".format(args.target_epsilon))
    print("-" * 20)
    
    print("Global rounds: {}".format(args.global_rounds))
    print("Local batch size train: {}".format(args.batch_size_train))
    print("Batch size test: {}".format(args.batch_size_test))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local epochs for new clients: {}".format(args.fine_tuning_epoch_new))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Beta1: {}".format(args.beta1))
    print("Beta2: {}".format(args.beta2))
    print("eps: {}".format(args.eps))
    print("Weight decay: {}".format(args.weight_decay))
    print("Learning rate decay: {}".format(args.learning_rate_decay))
    print("Learning rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("=" * 50)

    run(args)