import itertools
import subprocess
import threading
import time
import os
from pathlib import Path

import torch

# Define the path to main.py
# script_dir = Path("/data/FL-UTAustin/fednew/system")
script_dir = Path("/data/FedCLIP/system")
main_script = script_dir / "main.py"

# Check if GPUs are available
num_devices = torch.cuda.device_count()
# num_devices = 3

print(f'num_devices: {num_devices}')

if num_devices < 1:
    print("Please select a machine with at least 1 GPU.")
    quit()


dataset = "d47"
partition = "dir"
algo = "fdual"



# Define hyperparameter search space
learning_rates = [5e-5, 1e-5, 5e-6, 1e-6]
# learning_rates = [1e-6]

# batch_size_ref=(4 8 16 32)
# batch_size_ref=(4)
# batch_size_ref=(8)
# batch_size_ref=(16)
batch_size_ref=(32)

# distill_learning_rate = [1e-5, 1e-4, 1e-3]
# distill_learning_rate = [5e-5, 1e-5, 5e-6, 1e-6]
# distill_learning_rate = [5e-3, 1e-3, 5e-4, 1e-4]
distill_learning_rate = [5e-5]

# distill_epochs = [1, 2, 5]
distill_epochs = [1]

# distill_temp = [0.5, 1.0, 3.0]
distill_temp = [3.0]

# ref_data_fraction = [0.1, 0.2, 0.5, 1.0]
ref_data_fraction = [0.1]

seeds = [0]

# Define log directory
# log_dir = Path("/data/FL-UTAustin/fednew/logs/sumo_bev")
log_dir = Path(f"/data/FedCLIP/logs/{dataset}")
log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Track GPU availability
gpu_usage = {device: False for device in range(num_devices)}

# Track used hyperparameter combinations
used_combinations = set()


def run_script(lr, rbs, dlr, de, dt, rf, seed, device):
    """Runs the script with assigned parameters on a given GPU"""
    global gpu_usage

    job_name = f"{dataset}_{partition}_{algo}v_lr{lr}_rbs{rbs}_dlr{dlr}_de{de}_dt{dt}_rf{rf}_sd{seed}_all_pfl"
    output_file = log_dir / f"{job_name}.out"
    error_file = log_dir / f"{job_name}.err"

    command = [
        "python", str(main_script),
        "-data", str(dataset),
        "-algo", str(algo),
        "-gr", "50",
        "-did", str(device),  # Pass GPU ID as argument
        "-nc", "10",
        "-lbs", "32",
        "-lr", str(lr),
        "-rbs", str(rbs),
        "-distill_learning_rate", str(dlr),
        "-distill_epochs", str(de),
        "-distill_temp", str(dt),
        "-ref_data_fraction", str(rf),
        "--lora_rank", "2",
        "--lora_alpha", "16",
        "--lora_key_vision",
	    "--lora_query_vision",
	    "--lora_value_vision",
	    "--lora_outproj_vision",
	    "--lora_mlp_vision",
	    "--lora_head_vision",
        "-pfl",
        "-sd", str(seed)
    ]

    # Ensure `CUDA_VISIBLE_DEVICES` is set before launching `main.py`
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)  # Assign the correct GPU
    # env["CUDA_VISIBLE_DEVICES"] = '3'  # Assign the correct GPU

    print(f"[GPU {device}] Starting job: {job_name}")

    start_time = time.time()  # Record start time

    with output_file.open("w") as out, error_file.open("w") as err:
        # subprocess.run(command, stdout=out, stderr=err)
        subprocess.run(command, stdout=out, stderr=err, env=env)  # Pass modified environment

    end_time = time.time()  # Record end time
    elapsed_time = round(end_time - start_time, 2)  # Compute elapsed time

    # Append execution time summary to the log file
    with output_file.open("a") as out:
        out.write("\n===========================\n")
        out.write("Execution Time Summary:\n")
        out.write(f"Total Time: {elapsed_time} seconds\n")
        out.write("===========================\n")

    print(f"[GPU {device}] Completed job: {job_name} in {elapsed_time} seconds")

    # Mark GPU as free
    gpu_usage[device] = False


def get_available_gpu():
    """Find an available GPU and block if all are busy"""
    while True:
        for device, in_use in gpu_usage.items():
            if not in_use:
                gpu_usage[device] = True  # Mark GPU as busy
                return device
        time.sleep(10)  # Wait before checking again


# List to track threads
threads = []

# Generate all combinations of hyperparameters
for params in itertools.product(learning_rates, distill_learning_rate, distill_epochs, distill_temp, ref_data_fraction, seeds):
    # lr, seed = params
    lr, dlr, de, dt, rf, seed = params

    # Check if the combination has been used
    if params in used_combinations:
        continue
    used_combinations.add(params)

    # Get an available GPU
    device = get_available_gpu()

    # Start a thread to run the script
    thread = threading.Thread(target=run_script, args=(lr, dlr, de, dt, rf, seed, device))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All jobs completed.")
