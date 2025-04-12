#!/bin/bash
#SBATCH --array=0-11%3  # Replace <NUM_JOBS-1> with the total number of configurations minus 1
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

algo='floradattruemoe'

# Parameters arrays
moe_hidden_scales=(1 2 4)

ranks=(2)
alphas=(16)


learning_rates=(5e-5 1e-5 5e-6 1e-6)
# learning_rates=(1e-5 5e-6)                  #dir
# learning_rates=(1e-5 5e-6 1e-6)             # dir10
# learning_rates=(1e-5 5e-6 1e-6)             # dir001

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
total_configs=$(( ${#moe_hidden_scales[@]} * ${#ranks[@]} * ${#alphas[@]} * ${#learning_rates[@]} * ${#weight_decays[@]} ))

# Calculate the current configuration index from SLURM_ARRAY_TASK_ID
config_id=$SLURM_ARRAY_TASK_ID

# Derive the parameters for this job
hs_idx=$(( config_id % ${#moe_hidden_scales[@]} ))
config_id=$(( config_id / ${#moe_hidden_scales[@]} ))

r_idx=$(( config_id % ${#ranks[@]} ))
config_id=$(( config_id / ${#ranks[@]} ))

a_idx=$(( config_id % ${#alphas[@]} ))
config_id=$(( config_id / ${#alphas[@]} ))

lr_idx=$(( config_id % ${#learning_rates[@]} ))
config_id=$(( config_id / ${#learning_rates[@]} ))

wd_idx=$(( config_id % ${#weight_decays[@]} ))

# Get the actual parameter values
hs=${moe_hidden_scales[$hs_idx]}
r=${ranks[$r_idx]}
a=${alphas[$a_idx]}
lr=${learning_rates[$lr_idx]}
wd=${weight_decays[$wd_idx]}

# Define job name and log files
job_name="${dataset}_${partition}_${algo}v_lr${lr}_wd${wd}_r${r}_a${a}_hs${hs}_all_pfl"
output_file="${log_dir}/${job_name}.out"
error_file="${log_dir}/${job_name}.err"

# Redirect stdout and stderr
exec >"${output_file}" 2>"${error_file}"

echo "$PWD"
echo "Running with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, hs=${hs}"

# Run the command
nvidia-smi -L 
nvidia-smi --query-gpu=memory.total --format=csv 
nvidia-smi --query-gpu=compute_cap --format=csv 
nvidia-smi --query-gpu=power.max_limit --format=csv 
echo "GPU details saved to $output_file"

time python main.py -data ${dataset} \
    -algo ${algo} \
    -gr 100 \
    -did 0 \
    -nc 10 \
    -lbs 32 \
    -lr ${lr} \
    -wd ${wd} \
    --moe_hidden_scale ${hs} \
    --lora_rank ${r} \
    --lora_alpha ${a} \
    --lora_key_vision \
    --lora_query_vision \
    --lora_value_vision \
    --lora_outproj_vision \
    --lora_mlp_vision \
    --lora_head_vision \
    -pfl \
    -sd 0

echo "Submitted job $job_name with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, hs=${hs} at $(date)"
echo "Finished submitting all jobs at $(date)"
