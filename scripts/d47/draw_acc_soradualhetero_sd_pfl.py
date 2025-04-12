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

dataset = 'd47'

# partition = 'iid'
# partition = 'dir'
# partition = 'dir05'
# partition = 'dir10'
partition = 'dir001'

alg = 'fsoradualheterov'
# alg = 'fsoradualheterot'
# alg = 'fsoradualhetero2'

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
    
    plt.title(f'DTD | {partition} | {alg}', fontsize=fontsize, fontweight='bold')
    plt.xlabel('Round', fontsize=fontsize)
    plt.ylabel('Top-1 accuracy (%)', fontsize=fontsize)
    plt.legend(fontsize=fontsize_small, ncol=2, loc='lower right', bbox_to_anchor=(1, 0))
    plt.grid(True)
    
    plt.savefig(f'{dataset}_{partition}_{alg}_{r}_{rl}_{a}_{sl}_{gl}_sd_all_pfl.pdf', format='pdf', bbox_inches='tight')

    plt.close()


# lrs = ['5e-5', '1e-5']                    # fsoradualheterov dir 
lrs = ['5e-5']                            # fsoradualheterov dir001

# lrs = ['5e-5']                            # fsoradualheterot dir 
# lrs = ['5e-5']                            # fsoradualheterot dir001 


# gammas_local = ['0.001']            # fsoradualheterov dir 
gammas_local = ['0.01']             # fsoradualheterov dir001
# gammas_local = ['0.1']              # fsoradualheterov 
# gammas_local = ['1']                # fsoradualheterot  
# gammas_local = ['10']               # fsoradualheterot dir, dir001

# sls = ['1e-5']
sls = ['5e-5']
# sls = ['1e-4']

# wds = ['1', '9e-1', '8e-1', '7e-1', '6e-1', '5e-1', '4e-1', '3e-1', '2e-1', '1e-1', '1e-2', '1e-3', '0']
wds = ['0']

ranks = ['2']

ranks_local = ['4']

# alphas = ['2', '4', '8', '16', '32', '64']
# alphas = ['2']
# alphas = ['4']
# alphas = ['8'] 
alphas = ['16']
# alphas = ['32']
# alphas = ['64']

seeds = ['0', '1', '42']

# Create filenames and labels for combinations of lr and wd
filenames = []
labels = []

for sd in seeds: 
    for gl in gammas_local:
        for sl in sls:
            for lr in lrs:
                for wd in wds:
                    for r in ranks:
                        for rl in ranks_local:
                            for a in alphas:
                                filenames.append(f'../../logs/{dataset}/{dataset}_{partition}_{alg}_lr{lr}_wd{wd}_r{r}_rl{rl}_a{a}_sl{sl}_gl{gl}_sd{sd}_all_pfl.out')
                                labels.append(f'lr={lr}, wd={wd}, r={r}, rl={rl}, a={a}, sl={sl}, gl={gl}, sd={sd}')

colors = [blue_color, orange_color, green_color, red_color, purple_color, yellow_color, pink_color, brown_color, gray_color, olive_color, teal_color, black_color, cyan_color]

plot_multiple_accuracy_curves(filenames, labels, colors)



