#!/bin/bash
#SBATCH --array=0-1  # Replace <NUM_JOBS-1> with the total number of configurations minus 1
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --time=0-8:00:00
#SBATCH --mail-user='dphuong@iastate.edu'
#SBATCH --mail-type=END

# Dataset
dataset='p37'

# Partition
# partition='iid'
# partition='dir10'
partition='dir'
# partition='dir001'
# partition='shift'

algo='floradp'

# Parameters arrays
# target_epsilons=(0.2 2 20)
target_epsilons=(2)

ranks=(2)
alphas=(16)

# learning_rates=(5e-5 1e-5 5e-6 1e-6)
learning_rates=(1e-6 5e-7)                # dir
# learning_rates=(1e-5 5e-6 1e-6)           # dir10
# learning_rates=(1e-6)                     # dir001

weight_decays=(0)

# Define the directory where you want to store output and error files
log_dir="/work/LAS/jannesar-lab/dphuong/FedCLIP/logs/${dataset}"
mkdir -p $log_dir

echo "Loading modules"
module load miniconda3/22.11.1-hydt3qz
source activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora

cd /work/LAS/jannesar-lab/dphuong/FedCLIP/system
echo "$PWD"
echo "Started batch job at $(date)"

# Calculate total number of configurations
total_configs=$(( ${#target_epsilons[@]} * ${#ranks[@]} * ${#alphas[@]} * ${#learning_rates[@]} * ${#weight_decays[@]} ))

# Calculate the current configuration index from SLURM_ARRAY_TASK_ID
config_id=$SLURM_ARRAY_TASK_ID

# Derive the parameters for this job
te_idx=$(( config_id % ${#target_epsilons[@]} ))
config_id=$(( config_id / ${#target_epsilons[@]} ))

r_idx=$(( config_id % ${#ranks[@]} ))
config_id=$(( config_id / ${#ranks[@]} ))

a_idx=$(( config_id % ${#alphas[@]} ))
config_id=$(( config_id / ${#alphas[@]} ))

lr_idx=$(( config_id % ${#learning_rates[@]} ))
config_id=$(( config_id / ${#learning_rates[@]} ))

wd_idx=$(( config_id % ${#weight_decays[@]} ))

# Get the actual parameter values
te=${target_epsilons[$te_idx]}
r=${ranks[$r_idx]}
a=${alphas[$a_idx]}
lr=${learning_rates[$lr_idx]}
wd=${weight_decays[$wd_idx]}

# Define job name and log files
job_name="${dataset}_${partition}_${algo}t_lr${lr}_wd${wd}_r${r}_a${a}_te${te}_all_pfl"
output_file="${log_dir}/${job_name}.out"
error_file="${log_dir}/${job_name}.err"

echo "$PWD"
echo "Running with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, te=${te}"

# Run the command
nvidia-smi -L > $output_file
nvidia-smi --query-gpu=memory.total --format=csv >> $output_file
nvidia-smi --query-gpu=compute_cap --format=csv >> $output_file
nvidia-smi --query-gpu=power.max_limit --format=csv >> $output_file
echo "GPU details saved to $output_file"

time python main.py -data ${dataset} \
    -algo ${algo} \
    --target_epsilon ${te} \
    -gr 100 \
    -did 0 \
    -nc 10 \
    -lbs 32 \
    -lr ${lr} \
    -wd ${wd} \
    --lora_rank ${r} \
    --lora_alpha ${a} \
    --lora_key_text \
    --lora_query_text \
    --lora_value_text \
    --lora_outproj_text \
    --lora_mlp_text \
    --lora_head_text \
    -pfl \
    -sd 0

echo "Submitted job $job_name with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, te=${te} at $(date)"
echo "Finished submitting all jobs at $(date)"
