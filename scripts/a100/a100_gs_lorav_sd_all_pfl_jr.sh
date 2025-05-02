#!/bin/bash
# Dataset 
dataset='a100'

# Partition
# partition='iid'
# partition='dir10'
partition='dir'
# partition='dir001'

algo='flora'

ranks=(2)
# ranks=(4 8)
# alphas=(2 4 8 16 32 64)
alphas=(16)

# Define the directory where you want to store output and error files
log_dir="/data/FedCLIP/logs/${dataset}"
# log_dir="/scratch/bczq/FedCLIP/logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

echo "Loading modules"
# module load miniconda3/22.11.1-hydt3qz  # update this if necessary by "module spider conda"
# source activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora_pronto
# source activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora
# source activate /scratch/bczq/miniconda3/envs/flora

cd /data/FedCLIP/system
# cd /scratch/bczq/FedCLIP/system
echo "$PWD"
echo "Started batch job at $(date)"

# learning_rates=(5e-5 1e-5 5e-6 1e-6)
learning_rates=(1e-5)                   #dir
# learning_rates=(5e-5)                   #dir001

# weight_decays=(0 1e-3 1e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1)
weight_decays=(0)

join_ratio=(0.5)

# seeds=(1 42)
seeds=(0)
seeds=(1)
seeds=(42)

# Force the script to use GPU 
export CUDA_VISIBLE_DEVICES=3

for sd in "${seeds[@]}"; do
    for r in "${ranks[@]}"; do
        for a in "${alphas[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for wd in "${weight_decays[@]}"; do
                    for jr in "${join_ratio[@]}"; do
                        job_name="${dataset}_${partition}_${algo}v_lr${lr}_wd${wd}_r${r}_a${a}_sd${sd}_all_pfl_jr${jr}"
                        output_file="${log_dir}/${job_name}.out"
                        error_file="${log_dir}/${job_name}.err"

                        # Clear previous logs
                        > $output_file
                        > $error_file

                        echo "Running with algo=${algo}, lr=${lr}, sd=${sd}, jr=${jr}" | tee -a $output_file

                        # Capture GPU info before execution
                        {
                            echo "==========================="
                            echo "GPU Info:"
                            nvidia-smi -L
                            nvidia-smi --query-gpu=memory.total --format=csv
                            nvidia-smi --query-gpu=compute_cap --format=csv
                            nvidia-smi --query-gpu=power.max_limit --format=csv
                            echo "==========================="
                        } >> $output_file

                        echo "GPU details saved to $output_file" | tee -a $output_file
                        echo "Running training script..." | tee -a $output_file

                        # Run the command and measure execution time

                        {
                            START_TIME=$(date +%s)

                            time python main.py -data ${dataset} \
                                -algo ${algo} \
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
                                -rjr \
                                -jr ${jr} \
                                -CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES} \
                                -pfl \
                                -sd ${sd}\
                            >> $output_file 2>> $error_file

                            END_TIME=$(date +%s)
                            ELAPSED_TIME=$((END_TIME - START_TIME))

                            echo "===========================" >> $output_file
                            echo "Execution Time Summary:" >> $output_file
                            echo "Total Time: ${ELAPSED_TIME} seconds" >> $output_file
                            echo "===========================" >> $output_file

                        } 2>> $output_file &

                        echo "Started job ${job_name} at $(date)"
                    done
                done
            done
        done
    done
done

wait
echo "Finished submitting all jobs at $(date)"
