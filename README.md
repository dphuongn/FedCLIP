
## Operating System
Linux x86_64

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *flora*. 
```
conda env create -f env.yaml # for Linux or Windows with CUDA >= 12.1
conda activate flora
```

Or using micromamba:
```
micromamba env create -p /work/LAS/jannesar-lab/dphuong/.micromamba/envs/flora -f env_micromamba.yaml
```
(change 'dphuong' in '/work/LAS/jannesar-lab/dphuong/.micromamba/envs/flora' to your corresponding name folder)

then:
```
micromamba activate /work/LAS/jannesar-lab/dphuong/.micromamba/envs/flora
```

## Generating datasets

We provide **4** datasets: **Oxford-IIIT Pet (Pets)**,  **Describable Textures Dataset (DTD)**, **Oxford 102 Flower (Flowers)**, and **FGVC-Aircraft (Aircraft)**. They can be easy split into **IID** and **non-IID** version. For **non-IID**, we have practical setting (with hyperpameter for Dirichlet distribution $\beta$) and pathological setting (few-shot scenario). 

### Examples for **Pets**
- Total 10 clients, iid and balance scenario
    ```
    cd ./dataset
    python generate_pets.py 10 iid balance - - - - pfl
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.1$ 
    ```
    cd ./dataset
    python generate_pets.py 10 noniid - dir 0.1 - - pfl
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.01$ 
    ```
    cd ./dataset
    python generate_pets.py 10 noniid - dir 0.01 - - pfl



## Training and Evaluation

After generating and partitioning dataset for clients, we can run the training and evaluation. All codes corresponding to **FedDLP** and other baselines: **Local**, **FedCLIP**, **FLoRA**, and **Fed...** are stored in `./scripts`. Different folder corresponds with that specific dataset. The results will be saved in the `logs` folder.

### Examples on **DTD (d47)** for loraselectmost (select LoRA at most important layers (including head) to communicate)
```
bash ./scripts/d47/d47_gs_loraselectmostt_sd_all_pfl.sh         # for text encoder
bash ./scripts/d47/d47_gs_loraselectmostv_sd_all_pfl.sh        # for image encoder
```


## Parameters

| Parameter | Description |
| --------- | ----------- |
|`data`     | Dataset to use. Options: `pets (p37)`, `dtd (d47)` , `flowers (f102)`, `aircraft (a100)` (default: `pets`).|          
| `m`       | The base model. Options: `vit-b-32`, `vit-b-16`, `vit-l-14`, and `vit-l-14-336` (default: `vit-b-32`).|
| `algo`    | The training algorithm. Options: `fedloralocal (Local LoRA training)`, `fedclip (FedCLIP)`, `flora (FLoRA)`.|
| `gr`      | Number of communication rounds (default: `100`). |
| `jr`      | Ratio of participating clients per round (default: `1`). |
| `did`     | GPU device ID (default: `0`). |
| `nc`      | Number of clients (default: `10`). |
| `lbs`     | Training batch size (default: `32`). |
| `lr`      | Learning rate (default: `1e-5`). |
| `wd`      | Weight decay (default: `0`). |
| `pfl`     | For personalized federated learning if present, traditional federated learning otherwise. |
| `sd`      | The random seed (default: `0`). |


Feel free to change parameters to your desired experiments. If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper.
