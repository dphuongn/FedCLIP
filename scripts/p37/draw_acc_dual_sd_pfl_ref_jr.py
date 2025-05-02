import matplotlib.pyplot as plt
import re

blue_color = '#1f77b4'      
orange_color = '#ff7f0e'    
green_color = '#2ca02c'     
red_color = '#d62728'       
purple_color = '#9467bd'    

yellow_color = '#ffdd57'    
pink_color = '#e377c2'      
brown_color = '#8c564b'     
gray_color = '#7f7f7f'      
olive_color = '#bcbd22'  

teal_color = '#17becf'      
black_color = '#000000'     
lightblue_color = '#87ceeb' 
magenta_color = '#ff00ff'   
cyan_color = '#00ffff'

dataset = 'p37'

# partition = 'iid'
partition = 'dir'
# partition = 'dir05'
# partition = 'dir10'
# partition = 'dir001'

alg = 'fdualv'
# alg = 'fdualt'
# alg = 'flora2'

fontsize = 24 
fontsize_legend = 16
fontsize_small = 20
markersize = 8

def extract_rounds_and_accuracies(filename):
    rounds_and_accuracies = []
    with open(filename, 'r') as file:
        for line in file:
            if 'Round number' in line:
                round_number = int(re.search(r'Round number: (\d+)', line).group(1))
            if 'Averaged Test Accuracy' in line:
                accuracy = float(re.search(r'Averaged Test Accuracy: (\d+\.\d+)', line).group(1)) * 100
                rounds_and_accuracies.append((round_number, accuracy))
    return rounds_and_accuracies

def plot_multiple_accuracy_curves(filenames, labels, colors):
    plt.figure(figsize=(10, 6))
    
    for filename, label, color in zip(filenames, labels, colors):
        rounds_and_accuracies = extract_rounds_and_accuracies(filename)
        rounds, accuracies = zip(*rounds_and_accuracies)
        
        marker='o'
        fillstyle='none'
        
        plt.plot(rounds, accuracies, marker=marker, fillstyle=fillstyle, markersize=markersize, linewidth=3, color=color, label=label)
        
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.title(f'Pets | {partition} | {alg}', fontsize=fontsize, fontweight='bold')
    plt.xlabel('Round', fontsize=fontsize)
    plt.ylabel('Top-1 accuracy (%)', fontsize=fontsize)
    plt.legend(fontsize=fontsize_small, ncol=2, loc='lower right', bbox_to_anchor=(1, 0))
    plt.grid(True)
    
    plt.savefig(f'{dataset}_{partition}_{alg}_gs_all_pfl_ref_jr{jr}.pdf', format='pdf', bbox_inches='tight')

    plt.close()

# lrs = ['5e-5', '1e-5', '5e-6', '1e-6']
# lrs = ['5e-4', '1e-4', '5e-5', '1e-5', '5e-6', '1e-6']
# lrs = ['1e-4']
# lrs = ['5e-5']

lrs = ['1e-5', '5e-6']                        # florav dir
# lrs = ['5e-6']                        # florav dir

# lrs = ['1e-5']                        # florav dir001

wds = ['0']
ranks = ['2']
alphas = ['16']

# batch_size_ref = ['4', '8', '16', '32']
# batch_size_ref = ['16', '32']
batch_size_ref = ['32']

# distill_learning_rate = ['1e-5', '1e-4', '1e-3']
# distill_learning_rate = ['1e-5']
# distill_learning_rate = ['5e-5', '1e-5', '5e-6', '1e-6']
distill_learning_rate = ['5e-5']
# distill_learning_rate = ['0.005', '0.001', '0.0005', '0.0001']

distill_epochs = ['1']
# distill_epochs = ['1', '2']
# distill_epochs = ['2', '5']

# distill_temp = ['0.5', '1.0', '3.0']
distill_temp = ['3.0']
# distill_temp = ['1.0', '3.0']

# ref_data_fraction = ['0.1', '0.2', '0.5', '1.0']
ref_data_fraction = ['0.1']
# ref_data_fraction = ['1.0']

# Create filenames and labels for combinations of lr and wd
filenames = []
labels = []

# seeds = ['0', '1', '42']
seeds = ['0']

join_ratio = ['0.5']

for sd in seeds: 
    for lr in lrs:
        for rbs in batch_size_ref:
            for dlr in distill_learning_rate:
                for de in distill_epochs:
                    for dt in distill_temp:
                        for rf in ref_data_fraction:
                            for jr in join_ratio:
                                filenames.append(f'../../logs/{dataset}/{dataset}_{partition}_{alg}_lr{lr}_rbs{rbs}_dlr{dlr}_de{de}_dt{dt}_rf{rf}_sd{sd}_all_pfl_ref_jr{jr}.out')
                                labels.append(f'lr={lr}, rbs={rbs}, dlr={dlr}, de={de}, dt={dt}, rf={rf}, sd={sd}, jr={jr}')

# for sd in seeds: 
#     for lr in lrs:
#         filenames.append(f'../../logs/{dataset}/{dataset}_{partition}_{alg}_lr{lr}_sd{sd}_all_pfl.out')
#         labels.append(f'lr={lr}')

colors = [blue_color, orange_color, green_color, red_color, purple_color, yellow_color, pink_color, brown_color, gray_color, olive_color, teal_color, black_color, cyan_color]

plot_multiple_accuracy_curves(filenames, labels, colors)



