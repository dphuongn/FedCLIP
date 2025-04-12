#!/bin/bash
# Dataset 
dataset='p37'

# Partition
# partition='iid'
# partition='dir10'
# partition='dir'
partition='dir001'

algo='fedaa'

aa_b_rs=(1)

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

# learning_rates=(5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
# learning_rates=(5e-5)                           # dir
learning_rates=(5e-5)                           # dir001

# weight_decays=(0 1e-3 1e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1)
weight_decays=(0)

# seeds=(0 1 42)
seeds=(0)

for sd in "${seeds[@]}"; do
    for r in "${aa_b_rs[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for wd in "${weight_decays[@]}"; do
                job_name="${dataset}_${partition}_${algo}t_lr${lr}_wd${wd}_r${r}_sd${sd}_pfl"
                output_file="${log_dir}/${job_name}.out"
                error_file="${log_dir}/${job_name}.err"

                echo "$PWD"
                echo "Running with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, sd=${sd}"

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
                                -gr 100 \
                                -did 0 \
                                -nc 10 \
                                -lbs 32 \
                                -lr ${lr} \
                                -wd ${wd} \
                                --aa_bottleneck_reduction ${r} \
                                --aa_text \
                                -pfl \
                                -sd ${sd}\""

                echo "Submitting job with command: $sbatch_cmd"
                eval $sbatch_cmd

                echo "Submitted job $job_name with algo=${algo}, lr=${lr}, wd=${wd}, r=${r}, sd=${sd} at $(date)"
            done
        done
    done
done

echo "Finished submitting all jobs at $(date)"
