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

# alg = 'fmetav'
alg = 'fmetat'
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
    
    plt.savefig(f'{dataset}_{partition}_{alg}_gs_all_pfl.pdf', format='pdf', bbox_inches='tight')

    plt.close()

lrs = ['5e-05', '1e-05', '5e-06', '1e-06']
lrs = ['0.001', '0.005', '0.0001', '0.0005']
lrs = ['5e-05', '1e-05', '5e-06', '1e-06', '5e-07', '1e-07']
# lrs = ['5e-6']                        # florav dir
# lrs = ['5e-06']                        # florat dir

# lrs = ['5e-6']                        # florav dir001
# lrs = ['1e-6']                        # florat dir001

wds = ['0']
ranks = ['2']
alphas = ['16']

inner_steps = ['1', '3', '5']
# inner_steps = ['1']
# meta_inner_lr = ['1e-2', '2e-2', '5e-2']
# meta_inner_lr = ['0.01', '0.02', '0.05']
# meta_inner_lr = ['0.01']
meta_inner_lr = ['0.02']
# meta_inner_lr = ['0.01', '0.02']
# gumbel_temp = [1e-1, 5e-1, 1]
# gumbel_temp = ['0.1', '0.5', '1']
# gumbel_temp = ['0.1', '1']
# gumbel_temp = ['0.1']
gumbel_temp = ['1']
# meta_support_fraction = [3e-1, 5e-1]
meta_support_fraction = ['0.5']

# Create filenames and labels for combinations of lr and wd
filenames = []
labels = []

# seeds = ['0', '1', '42']
seeds = ['0']

for sd in seeds: 
    for lr in lrs:
        for i_s in inner_steps:
            for m_i_l in meta_inner_lr:
                for g_t in gumbel_temp:
                    for m_s_f in meta_support_fraction:
                        filenames.append(f'../../logs/{dataset}/{dataset}_{partition}_{alg}_lr{lr}_is{i_s}_mil{m_i_l}_gt{g_t}_msf{m_s_f}_sd{sd}_all_pfl.out')
                        labels.append(f'lr={lr}, is={i_s}, mil={m_i_l}, gt={g_t}, msf={m_s_f}')

colors = [blue_color, orange_color, green_color, red_color, purple_color, yellow_color, pink_color, brown_color, gray_color, olive_color, teal_color, black_color, cyan_color]

plot_multiple_accuracy_curves(filenames, labels, colors)



