#!/bin/bash
# Dataset 
dataset='p37'

# Partition
# partition='iid'
# partition='dir10'
partition='dir'
# partition='dir001'

algo='fsoradualheterothaw'

# gammas_local=(0.1 1)
# gammas_local=(0.01 0.1)
gammas_local=(0.01)             # dir
# gammas_local=(0.01)             # dir001

# gammas_global=(0.1 10)          # dir
# gammas_global=(0.01 100)        # dir

# sls=(1e-5)
sls=(5e-5)
# sls=(1e-4)

ranks=(2)
ranks_local=(4)

# alphas=(2 4 8 16 32 64)
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

learning_rates=(5e-5 1e-5 5e-6 1e-6)
# learning_rates=(1e-5)                   # dir
# learning_rates=(1e-5)                   # dir001

# weight_decays=(0 1e-3 1e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1)
weight_decays=(0)

# seeds=(0 1 42)
seeds=(0)

for sd in "${seeds[@]}"; do
    for gl in "${gammas_local[@]}"; do
        for sl in "${sls[@]}"; do
            for rl in "${ranks_local[@]}"; do
                for r in "${ranks[@]}"; do
                    for a in "${alphas[@]}"; do
                        for lr in "${learning_rates[@]}"; do
                            for wd in "${weight_decays[@]}"; do
                                job_name="${dataset}_${partition}_${algo}v_lr${lr}_wd${wd}_r${r}_rl${rl}_a${a}_sl${sl}_gl${gl}_sd${sd}_all_pfl"
                                output_file="${log_dir}/${job_name}.out"
                                error_file="${log_dir}/${job_name}.err"

                                echo "$PWD"
                                echo "Running with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, sl=${sl}, rl=${rl}, gl=${gl}, sd=${sd}"

                                sbatch_cmd="sbatch --job-name=$job_name \
                                    --partition=nova \
                                    --gres=gpu:a100:1 \
                                    --nodes=1 \
                                    --mem=50G \
                                    --time=0-8:00:00 \
                                    --mail-user='dphuong@iastate.edu' \
                                    --mail-type=END \
                                    --output=$output_file \
                                    --error=$error_file \
                                    --wrap=\"nvidia-smi -L > $output_file && \
                                            nvidia-smi --query-gpu=memory.total --format=csv >> $output_file && \
                                            nvidia-smi --query-gpu=compute_cap --format=csv >> $output_file && \
                                            nvidia-smi --query-gpu=power.max_limit --format=csv >> $output_file && \
                                            echo 'GPU details saved to $output_file' && \
                                            time python main.py -data ${dataset} \
                                                -algo ${algo} \
                                                --sparse_lambda ${sl} \
                                                --gamma_local ${gl} \
                                                -gr 100 \
                                                -did 0 \
                                                -nc 10 \
                                                -lbs 32 \
                                                -lr ${lr} \
                                                -wd ${wd} \
                                                --lora_rank ${r} \
                                                --lora_rank_local ${rl} \
                                                --lora_alpha ${a} \
                                                --lora_key_vision \
                                                --lora_query_vision \
                                                --lora_value_vision \
                                                --lora_outproj_vision \
                                                --lora_mlp_vision \
                                                --lora_head_vision \
                                                -pfl \
                                                -sd ${sd}\""

                                echo "Submitting job with command: $sbatch_cmd"
                                eval $sbatch_cmd

                                echo "Submitted job $job_name with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, a=${a}, sl=${sl}, rl=${rl}, gl=${gl}, sd=${sd} at $(date)"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Finished submitting all jobs at $(date)"
