#!/bin/bash
# Dataset 
dataset='p37'

# Partition
# partition='iid'
# partition='dir10'
partition='dir'
# partition='dir001'

algo='fdualnogate'

ranks=(2)
alphas=(16)

# batch_size_ref=(4 8 16 32)
# batch_size_ref=(4)
# batch_size_ref=(8)
# batch_size_ref=(16)
batch_size_ref=(32)

# distill_learning_rate=(1e-5 1e-4 1e-3)
# distill_learning_rate=[5e-5 1e-5 5e-6 1e-6)
# distill_learning_rate=(5e-3 1e-3 5e-4 1e-4)
distill_learning_rate=(5e-5)
# distill_learning_rate=(1e-5)
# distill_learning_rate=(5e-6)
# distill_learning_rate=(1e-6)

# distill_epochs=(1 2 5)
distill_epochs=(1)
# distill_epochs=(2)
# distill_epochs=(5)

# distill_temp=(0.5 1.0 3.0)
# distill_temp=(1.0)
distill_temp=(3.0)

# ref_data_fraction=(0.1 0.5)
ref_data_fraction=(0.1)

# Define the directory where you want to store output and error files
log_dir="/work/LAS/jannesar-lab/dphuong/FedCLIP/logs/${dataset}"
# log_dir="/scratch/bczq/FedCLIP/logs/${dataset}"

# Create the directory if it doesn't exist
mkdir -p $log_dir

echo "Loading modules"
# module load micromamba
# eval "$(micromamba shell hook --shell=bash)"
# micromamba activate /work/LAS/jannesar-lab/dphuong/.micromamba/envs/flora

cd /work/LAS/jannesar-lab/dphuong/FedCLIP/system
# cd /scratch/bczq/FedCLIP/system
echo "$PWD"
echo "Started batch job at $(date)"

# learning_rates=(5e-5 1e-5 5e-6 1e-6)

learning_rates=(1e-5)                   # dir
# learning_rates=(5e-6)                   # dir

# learning_rates=(1e-5)                   # dir001

# weight_decays=(0 1e-3 1e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1)
weight_decays=(0)

# seeds=(1 42)
seeds=(0)
seeds=(1)
seeds=(42)

for sd in "${seeds[@]}"; do
    for rbs in "${batch_size_ref[@]}"; do
        for dlr in "${distill_learning_rate[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for de in "${distill_epochs[@]}"; do
                    for dt in "${distill_temp[@]}"; do
                        for rf in "${ref_data_fraction[@]}"; do
                            job_name="${dataset}_${partition}_${algo}v_lr${lr}_rbs${rbs}_dlr${dlr}_de${de}_dt${dt}_rf${rf}_sd${sd}_all_pfl"
                            output_file="${log_dir}/${job_name}.out"
                            error_file="${log_dir}/${job_name}.err"

                            # Clear previous logs
                            > $output_file
                            > $error_file

                            echo "Running with algo=${algo}, lr=${lr}, rbs=${rbs}, dlr=${dlr}, de=${de}, dt=${dt}, rf=${rf}, sd=${sd}" | tee -a $output_file
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
                                    -wd 0 \
                                    --lora_rank 2 \
                                    --lora_alpha 16 \
                                    -rbs ${rbs} \
                                    -distill_learning_rate ${dlr} \
                                    -distill_epochs ${de} \
                                    -distill_temp ${dt} \
                                    -ref_data_fraction ${rf} \
                                    --lora_key_vision \
                                    --lora_query_vision \
                                    --lora_value_vision \
                                    --lora_outproj_vision \
                                    --lora_mlp_vision \
                                    --lora_head_vision \
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
done

wait
echo "Finished all jobs at $(date)"
