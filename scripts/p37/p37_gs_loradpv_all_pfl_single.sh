#!/bin/bash

# Slurm directives
#SBATCH --job-name=all_runs
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --time=0-8:00:00
#SBATCH --mail-user='dphuong@iastate.edu'
#SBATCH --mail-type=END
#SBATCH --output=/scratch/bczq/FedCLIP/logs/p37/all_runs.out
#SBATCH --error=/scratch/bczq/FedCLIP/logs/p37/all_runs.err

# Dataset 
dataset='p37'

# Partition
partition='dir'

algo='floradp'

# target_epsilons=(0.2 2 20)
target_epsilons=(2)

ranks=(2)
alphas=(16)

# Define the directory where you want to store output and error files
log_dir="/work/LAS/jannesar-lab/dphuong/FedCLIP/logs/${dataset}"
# log_dir="/scratch/bczq/FedCLIP/logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

echo "Loading modules"
module load miniconda3/22.11.1-hydt3qz  # update this if necessary by "module spider conda"
# source activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora_pronto
source activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora
# source activate /scratch/bczq/miniconda3/envs/flora

cd /work/LAS/jannesar-lab/dphuong/FedCLIP/system
# cd /scratch/bczq/FedCLIP/system
echo "$PWD"
echo "Started batch job at $(date)"

# learning_rates=(5e-5 1e-5 5e-6 1e-6)
learning_rates=(1e-5 5e-6)                  #dir
# learning_rates=(1e-5 5e-6 1e-6)             # dir10
# learning_rates=(1e-5 5e-6 1e-6)             # dir001
weight_decays=(0)

for te in "${target_epsilons[@]}"; do
    for r in "${ranks[@]}"; do
        for a in "${alphas[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for wd in "${weight_decays[@]}"; do
                    job_name="${dataset}_${partition}_${algo}v_lr${lr}_wd${wd}_r${r}_a${a}_te${te}_all_pfl"
                    output_file="${log_dir}/${job_name}.out"
                    error_file="${log_dir}/${job_name}.err"

                    echo "$PWD"
                    echo "Running with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, te=${te}"

                    # Save GPU details to the output file
                    nvidia-smi -L > $output_file
                    nvidia-smi --query-gpu=memory.total --format=csv >> $output_file
                    nvidia-smi --query-gpu=compute_cap --format=csv >> $output_file
                    nvidia-smi --query-gpu=power.max_limit --format=csv >> $output_file
                    echo "GPU details saved to $output_file"

                    # Run the main Python script
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
                        --lora_key_vision \
                        --lora_query_vision \
                        --lora_value_vision \
                        --lora_outproj_vision \
                        --lora_mlp_vision \
                        --lora_head_vision \
                        -pfl \
                        -sd 0 \
                        >> $output_file 2>> $error_file

                    echo "Finished running ${job_name} at $(date)"
                done
            done
        done
    done
done

echo "Finished all jobs at $(date)"
