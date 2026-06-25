# %% [markdown]
# # Bias Analysis - QD vs. Decision Trees
# #### Code by Catalina Jaramillo and M Charity
# ---
# 
# ### Experiments
# [ ] - Evolving neural network weights
# 
# [ ] - Evolving decision trees
# 
# ### Datasets
# 
# 
# 
# ### Features
# - gender + age --> promotion prediction
# - gender + race --> income prediction
# 
# ### Results summary
# 
# 
# ### Other Notes
#  

# %% [markdown]
# ---- 
# 
# ### CODE SETUP

# %%
# imports

import math
import random
import itertools
import inspect
import time
import json

from tqdm import tqdm

# data proc
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set threshold to infinity to print the entire array
np.set_printoptions(threshold=np.inf) 

# pyribs
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter, EvolutionStrategyEmitter, GeneticAlgorithmEmitter, IsoLineEmitter
from ribs.schedulers import Scheduler
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap     # be sure to install shapely for this to work


# %%
# import external classes and files
from EvoDecisionTree import EvoDecTree, EvoNode
from EvoNeuralNet import EvoNN

# %% [markdown]
# ### UTILITY FUNCTIONS


# %%
# DIRECTORY SETUP
HEAD_DIR = "6-17"
ARCHIVE_DIR = f"{HEAD_DIR}/archive"
WEIGHTS_DIR = f"{HEAD_DIR}/weights"
RESULTS_DIR = f"{HEAD_DIR}/results"
GRID_DIR = f"{HEAD_DIR}/grid"


## EXPERIMENT SETTINGS
EXPERIMENT = "BOTH"
EXP_RANGE = [1]
EXP_ITERATIONS = 5000
DATASETS = ['synthetic']


AX = 10     # archve dimensions

# DATASETS = ['adult','german','compass']
# DATASETS = ['compass','adult']


# %%
# Combine lambdas for filtering rows in a DataFrame
def combine_lambdas(keys,l_dict):
    return lambda row: all(l_dict[k](row) for k in keys)

def get_combo_data(n,c,data,l_dict):
    ''' Returns a new dataset based on the given features 
        data is the original dataset
        c is the abbreviated combo
        l_dict is a dictionary of lambda functions to be used for filtering based on the features pairings
            - ex. ['f','p','y'] will filter for rows'''
    
    match = combine_lambdas(c,l_dict)
    m_data = data[data.apply(match,axis=1)]
    if m_data.shape[0] > 0:
        return m_data.sample(n)
    else:
        return pd.DataFrame()

# dataset sample creator
def unbiased_dataset(n,data,feats,l_dict):
    ''' Creates an unbiased dataset with all combinations of features the same size '''

    # Generate all combinations
    combos = list(itertools.product(*feats))
    combos = [list(comb) for comb in combos]

    # sample based on every combination
    new_data = pd.DataFrame()
    for c in combos:
        x = get_combo_data(n,c,data,l_dict)
        if x.shape[0] > 0:
            new_data = pd.concat([new_data, x])

    return new_data

def bias_n(n_bias, combo):
    ''' Returns the n value for the occurrence of a bias value '''
    for k,v in n_bias.items():
        if k == 'default':
            continue
        if all([i in combo for i in k]):        # has all the bias values
            return v

    return n_bias['default']

def bias_dataset(n_bias,data,feats,l_dict):
    ''' Creates a biased dataset with all combinations of features the same size for particular feature values '''

    # Generate all combinations
    combos = list(itertools.product(*feats))
    combos = [list(comb) for comb in combos]

    # sample based on every combination
    new_data = pd.DataFrame()
    for c in combos:
        # get the n value based on the bias in the dataset combo
        n = bias_n(n_bias,c)

        # get the matching data
        x = get_combo_data(n,c,data,l_dict)
        if x.shape[0] > 0:
            new_data = pd.concat([new_data, x])

    return new_data


# %%
def arx_vis(arx, x_prob, y_prob, a, b, scaled=False):
    plt.figure(figsize=(a, b))
    if scaled:
        grid_archive_heatmap(arx, vmin=0, vmax=1)
    else:
        grid_archive_heatmap(arx)
    plt.title("Accuracy")
    plt.xlabel(x_prob)
    plt.ylabel(y_prob)
    plt.zlabel()
    plt.show()

def arx_vis_out(arx, x_prob, y_prob, a, b, name, scaled=True):
    plt.figure(figsize=(a, b))
    if scaled:
        grid_archive_heatmap(arx, vmin=0, vmax=1)
    else:
        grid_archive_heatmap(arx)
    plt.title("Accuracy")
    plt.xlabel(x_prob)
    plt.ylabel(y_prob)
    plt.savefig(name)
    plt.close()


# %% [markdown]
# ---
# ### QD EXPERIMENT: Evolving NN Weights w/ CMA-ME
# 
# Rewriting Pyribs was a pain in the ASS :( why would anyone change the framework definition?!

# %%
def getArxSubInd(arx, xr=(4,6), yr=(4,6), zr=(4,6), dims=(AX,AX,AX)):
    ''' Returns a subset of the archive based on the x and y indices '''
    i = list(range(dims[0]*dims[1]*dims[2]))
    i = np.array(i).reshape(dims)
    #print(i)

    # get only the indices in between the xr and yr range
    j = i[xr[0]:xr[1], yr[0]:yr[1], zr[0]:zr[1]]
    return j.flatten().tolist()
    

# %%
a = 64
b = 32
c = 1

def batchFitNN(zs, X, y, comps):
    ''' Batch generation of vectors for use in the CMA-ME experiment with NNs'''
    global a, b, c

    fs = []
    cs = []

    for z in zs:
        nn = EvoNN(z)
        nn.set_params(a=a, b=b, c=c)
        nn.set_weights(z)

        f,c = nn.run(X, y, comps)
        fs.append(f)
        cs.append(c)
    return np.array(fs), np.array(cs)

def nnME(X,y, comparisons,iterations=100000, dname=""):
    ''' CMA-ME experiment using vector-based NNs model'''
    global a, b, c

    rang = [[0, 2], [0, 2], [0,2]]
    dims = [AX,AX,AX]
    s = ((X.shape[1]+1)*a)+((a+1)*b)+((b+1)*1)

    archive = GridArchive(solution_dim=s, ranges=rang, dims=dims)
    emitter1 = GaussianEmitter(archive, sigma=0.1, batch_size=32,
                              x0 = np.random.normal(scale=1, size=s))
    emitter2 = EvolutionStrategyEmitter(archive, sigma0=0.1, batch_size=32,
                              x0 = np.random.normal(scale=1, size=s))
    emitters = [emitter1, emitter2]
    opt = Scheduler(archive, emitters)       # optimizer was renamed to scheduler for some reason
    #opt.set_verbosity(1)

    start_time = time.time()
    solutions = []
    with tqdm(total=iterations) as pbar:
        for itr in range(iterations):
            pbar.set_description(f"[NN] Iteration {itr} [{dname}] - {len(archive)} / {archive.cells} elites")
            sols = opt.ask()
        
            # Reshape and normalize the image and pass it through the network.
            zs = sols
            objs, bcs = batchFitNN(zs, X, y, comparisons)
            
            # print(bcs)
            # print(f"objs: {objs} | bcs: {bcs}")
            # if itr%20 == 0: print (f"objs: {objs[0]} | bcs: {bcs[0]}")

            # assert not np.any(bcs > 2), f"Behavioral characteristic out of bounds )"

            # ignore invalid bcs
            # objs = []
            # bcs = []
            # for i in range(len(bcs)):
            #     if np.any(bcs[i] >= 0.5) and np.any(bcs[i] <= 2):
            #         objs.append(os[i])
            #         bcs.append(bs[i])

            # VALID MASK
            valid = np.all((bcs >= 0) & (bcs <= 2), axis=1)

            objs[~valid] = 0
            bcs[~valid] = -1.0

            # Mark invalid rows so they are ignored by the archive.
            # (Most pyribs versions treat NaN objective/BC as “do not insert”.)
            # objs[~valid] = np.nan
            # bcs[~valid] = np.nan

            opt.tell(objs, bcs)
            #print(str(len(archive.as_pandas()['index_0'].values)))
        
            if itr % 1000 == 0:
                print(f"[NN] Iteration {itr} complete after {(time.time() - start_time)/60} m")
                print(str(len(archive.data(return_type='pandas')['measures_0'].values)))
                solutions.append(str(len(archive.data(return_type='pandas')['measures_0'].values)))
                if len(solutions)>50 and solutions[-1] == solutions[-50]:
                # if len(solutions)>10 and solutions[-1] == solutions[-10]:
                    print('stagnant')
                    break


            # if itr % 1000 == 0:
            #     #arx_vis(archive, comparisons[0], comparisons[1], 4,3)
            #     arx_vis_out(archive, comparisons[0], comparisons[1], 8,6, f"{GRID_DIR}/NN/nn_exp_{dname}-_{itr}.png")

            pbar.update(1)


    # export
    arx = archive.data(return_type='pandas')
    # arx.to_pickle(f"{ARCHIVE_DIR}/NN/nn_exp_{dname}.pkl")

    # for models in the square (fair zone)
    square_i = getArxSubInd(arx, xr=(4,6), yr=(4,6), zr=(4,6), dims=(AX,AX,AX))
    square = arx.loc[arx['index'].isin(square_i)]
        
    best_model_unbiased_index = square['objective'].idxmax() if len(square) > 0 else None
    best_model_index = arx['objective'].idxmax()
    
    best_model_weights = arx.loc[best_model_index].iloc[5:].values if best_model_index is not None else None
    best_model_unbiased_weights = arx.loc[best_model_unbiased_index].iloc[5:].values if best_model_unbiased_index is not None else None

    # arx_vis_out(archive, comparisons[0], comparisons[1], 8,6, f"{GRID_DIR}/NN/nn_exp_{dname}_FINAL[0,1].png")
    # arx_vis_out(archive, comparisons[1], comparisons[2], 8,6, f"{GRID_DIR}/NN/nn_exp_{dname}_FINAL[1,2].png")
    # arx_vis_out(archive, comparisons[0], comparisons[2], 8,6, f"{GRID_DIR}/NN/nn_exp_{dname}_FINAL[0,2].png")

    return best_model_weights, best_model_unbiased_weights, square, arx, best_model_index, best_model_unbiased_index


# %% [markdown]
# ---
# ### DT EXPERIMENT: Evolving Decision Trees w/ CMA-ME

# %%
def batchFitDT(zs, X, y, comps):
    ''' Batch generation of vectors for use in the CMA-ME experiment with DTs'''
    X_np = X.to_numpy()
    y_np = y.to_numpy().flatten()
    
    fs = []
    cs = []
    for z in zs:
        dt = EvoDecTree(X_np,y_np)
        #print(z)
        dt.build_tree_from_vec(z)

        #print(dt)      # show tree

        f,c = dt.run(X, X_np, y_np, comps)
        fs.append(f)
        cs.append(c)

    return np.array(fs), np.array(cs)

## --- CUSTOM EMITTERS FOR INTEGER-ONLY SOLUTIONS ---
class IntegerBoundedGaussian(GaussianEmitter):
    def ask(self):
        # Get the samples from the GaussianEmitter
        solutions = super().ask()
        # Round to nearest integer
        solutions = np.rint(solutions)
        # Clip values to [0, 100]
        solutions = np.clip(solutions, 0, 1000)
        return solutions.astype(int)

class IntegerBoundedEvolutionStrategy(EvolutionStrategyEmitter):
    def ask(self):
        # Get the samples from the EvolutionStrategyEmitter
        solutions = super().ask()
        # Round to nearest integer
        solutions = np.rint(solutions)
        # Clip values to [0, 100]
        solutions = np.clip(solutions, 0, 1000)
        return solutions.astype(int)

def dtME(X,y, comparisons, iterations=100000, dname=""):
    ''' CMA-ME experiment using vector-based Decision Tree model'''
    rang = [[0, 2], [0, 2], [0,2]]
    dims = [AX,AX,AX]
    n = X.shape[1]
    s = 2**12      # decision tree vector size approximation
    print(f"DT vector size: {s}")
    print(f"X shape: {X.shape}")

    archive = GridArchive(solution_dim=s, ranges=rang, dims=dims)
    emitter2 = IntegerBoundedGaussian(archive, sigma=100.0, batch_size=32,
                                x0=np.full(s, 500))
    emitter3 = IntegerBoundedEvolutionStrategy(archive, sigma0=100.0, batch_size=32,
                                x0=np.full(s, 500))
    emitters = [emitter2, emitter3]
    opt = Scheduler(archive, emitters)
    #opt.set_verbosity(1)


    start_time = time.time()
    solutions = []
    with tqdm(total=iterations) as pbar:
        for itr in range(iterations):
            pbar.set_description(f"[DT] Iteration {itr} [{dname}] - {len(archive)} / {archive.cells} elites")
            sols = opt.ask()
        
            # Reshape and normalize the image and pass it through the network.
            zs = sols
            zs = [[min(max(1,int(i)),2**32) for i in z] for z in zs]      # convert to int and avoid 0s
            objs, bcs = batchFitDT(zs, X, y, comparisons)

            # print(bs)
            # print(f"objs: {os} | bcs: {bs}")
            # if itr%20 == 0: print (f"objs: {os[0]} | bcs: {bs[0]}")

            # objs = []
            # bcs = []
            # for i in range(len(bcs)):
            #     if np.any(bcs[i] >= 0.5) and np.any(bcs[i] <= 2):
            #         objs.append(os[i])
            #         bcs.append(bs[i])

            # VALID MASK
            valid = np.all((bcs >= 0) & (bcs <= 2), axis=1)

            # Mark invalid rows so they are ignored by the archive.
            # (Most pyribs versions treat NaN objective/BC as “do not insert”.)
            objs[~valid] = 0
            bcs[~valid] = -1.0

            opt.tell(objs, bcs)
            #print(str(len(archive.as_pandas()['index_0'].values)))
        
            if itr % 1000 == 0:
                print(f"[DT] Iteration {itr} complete after {(time.time() - start_time)/60} m")
                print(str(len(archive.data(return_type='pandas')['measures_0'].values)))
                solutions.append(str(len(archive.data(return_type='pandas')['measures_0'].values)))
                if len(solutions)>50 and solutions[-1] == solutions[-50]:
                # if len(solutions)>10 and solutions[-1] == solutions[-10]:
                    print('stagnant')
                    break

            # if itr % 1000 == 0:
            #     #arx_vis(archive, comparisons[0], comparisons[1], 4,3)
            #     arx_vis_out(archive, comparisons[0], comparisons[1], 8,6, f"{GRID_DIR}/DT/dt_exp_{dname}_{itr}.png")

            pbar.update(1)

    # export
    arx = archive.data(return_type='pandas')
    # arx.to_pickle(f"{ARCHIVE_DIR}/DT/dt_exp_{dname}.pkl")

    # for models in the square (fair zone)
    square_i = getArxSubInd(arx, xr=(4,6), yr=(4,6), zr=(4,6), dims=(AX,AX,AX))
    square = arx.loc[arx['index'].isin(square_i)]
        
    best_model_unbiased_index = square['objective'].idxmax() if not square.empty else None
    best_model_index = arx['objective'].idxmax()
    
    best_model_weights = arx.loc[best_model_index].iloc[5:].values if best_model_index is not None else None
    best_model_unbiased_weights = arx.loc[best_model_unbiased_index].iloc[5:].values if best_model_unbiased_index is not None else None

    # arx_vis_out(archive, comparisons[0], comparisons[1], 8,6, f"{GRID_DIR}/DT/dt_exp_{dname}_FINAL[0,1].png")
    # arx_vis_out(archive, comparisons[1], comparisons[2], 8,6, f"{GRID_DIR}/DT/dt_exp_{dname}_FINAL[1,2].png")
    # arx_vis_out(archive, comparisons[0], comparisons[2], 8,6, f"{GRID_DIR}/DT/dt_exp_{dname}_FINAL[0,2].png")

    return best_model_weights, best_model_unbiased_weights, square, arx, best_model_index, best_model_unbiased_index


# %% [markdown]
# ### Run Experiments

# %%
# %%
# IMPORT REAL DATASETS
### the adult, german, compass, and law datasets ###

# load individual dataset
synthetic = {}

comp_set = {
    'synthetic':['[GENDER] Male | Female','[RACE] Other | White','[AGE] <50 | 50+'], 
}



synthetic[1] = []
for p in ['X','y']:
    synthetic[1].append(pd.read_csv(f"./synthetic_hiring_bias_dataset_{p}.csv"))



# %%
# create directories if not there yet
import os
if not os.path.exists(ARCHIVE_DIR):
    os.makedirs(ARCHIVE_DIR)
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(GRID_DIR):
    os.makedirs(GRID_DIR)

for d in ['NN', 'DT']:
    if not os.path.exists(f"{ARCHIVE_DIR}/{d}"):
        os.makedirs(f"{ARCHIVE_DIR}/{d}")
    if not os.path.exists(f"{WEIGHTS_DIR}/{d}"):
        os.makedirs(f"{WEIGHTS_DIR}/{d}")
    if not os.path.exists(f"{RESULTS_DIR}/{d}"):
        os.makedirs(f"{RESULTS_DIR}/{d}")
    if not os.path.exists(f"{GRID_DIR}/{d}"):
        os.makedirs(f"{GRID_DIR}/{d}")

# %%
# experiment exporter

def export_weights(weights, test, name):
    ''' Export the weights to a file (csv) '''
    with open(f"{WEIGHTS_DIR}/{test}/{name}.csv", "w") as f:
        pd.DataFrame(weights).to_csv(f, index=False, header=False)


def export_results(results, test, name):
    ''' Export the results to a file '''
    with open(f"{RESULTS_DIR}/{test}/{name}.txt", "w") as f:
        for result in results:
            f.write(f"{result}\n")


def nn_exp(d, i, iterations=100000):
    ''' Run the NN experiment on the given dataset 
        d is the dataset name
        i is the dataset index
        iterations is the number of iterations to run the experiment for
    
    '''


    print("="*30)
    print(f"\nRunning NN experiment {i} on {d} dataset\n")
    print("="*30)
    X = eval(d)[i][0]
    y = eval(d)[i][1]

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler=MinMaxScaler()
    X_train= pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    comps = comp_set[d]

    best_model_weights, best_model_unbiased_weights, square, arx, best_model_index, best_model_unbiased_index = nnME(X,y, comps, iterations=iterations, dname=f"{d}-{i}")
     # print the best model weights
    # print(f"Best model weights: {np.array2string(best_model_weights)}")
    # print(f"Best unbiased model weights: {np.array2string(best_model_unbiased_weights)}")
    print(f"Best model index: {best_model_index}")
    print(f"Best unbiased model index: {best_model_unbiased_index}")

    # save the model
    date_time = time.strftime("%Y-%m-%d")
    export_weights(best_model_weights, "NN", f"nn-{d}_{i}-[{date_time}]")
    np.save(f"{WEIGHTS_DIR}/NN/{d}_{i}-[{date_time}].npy", best_model_weights)
    export_weights(best_model_unbiased_weights, "NN", f"nn-{d}_{i}_unbiased-[{date_time}]")
    np.save(f"{WEIGHTS_DIR}/NN/{d}_{i}_unbiased-[{date_time}].npy", best_model_unbiased_weights)


    # export the results
    res = []
    res.append(f"== DATA: {d.upper()} {i} ==")
    res.append(f"Iterations: {EXP_ITERATIONS}")
    res.append(f"Best model index: {best_model_index}")
    res.append(f"Best unbiased model index: {best_model_unbiased_index}")
    res.append(f"Best model weights: {np.array2string(best_model_weights)}")
    res.append(f"Best unbiased model weights: {np.array2string(best_model_unbiased_weights)}")
    export_results(res, "NN", f"nn-{d}_{i}-[{date_time}]")

    # save the archive
    arx.to_pickle(f"{ARCHIVE_DIR}/NN/nn-{d}_{i}-[{date_time}].pkl")


def dt_exp(d, i, iterations=100000):
    print("="*30)
    print(f"\nRunning DT experiment {i} on {d} dataset\n")
    print("="*30)

    X = eval(d)[i][0]
    y = eval(d)[i][1]

    comps = comp_set[d]

    best_model_vec, best_model_unbiased_vec, square, arx, best_model_index, best_model_unbiased_index = dtME(X,y, comps, iterations=iterations, dname=f"{d}-{i}")
    
    print(f"Best model index: {best_model_index}")
    print(f"Best unbiased model index: {best_model_unbiased_index}")
    # print(f"Best model vector: {str(best_model_vec)}")
    # print(f"Best unbiased model vector: {str(best_model_unbiased_vec)}")


    # save the model
    date_time = time.strftime("%Y-%m-%d")
    export_weights(best_model_vec, "DT", f"dt-{d}_{i}-[{date_time}]")
    np.save(f"{WEIGHTS_DIR}/DT/{d}_{i}-[{date_time}].npy", best_model_vec)
    #with open(f"{WEIGHTS_DIR}/DT/{d}_{i}-[{date_time}].csv", "w") as f:
    #    f.write(f"{best_model_vec}\n")

    export_weights(best_model_unbiased_vec, "DT", f"dt-{d}_{i}_unbiased-[{date_time}]")
    np.save(f"{WEIGHTS_DIR}/DT/{d}_{i}_unbiased-[{date_time}].npy", best_model_unbiased_vec)
    #with open(f"{WEIGHTS_DIR}/DT/{d}_{i}_unbiased-[{date_time}].csv", "w") as f:
    #    f.write(f"{best_model_unbiased_vec}\n")

    # export the results
    res = []
    res.append(f"== DATA: {d.upper()} {i} ==")
    res.append(f"Iterations: {EXP_ITERATIONS}")
    res.append(f"Best model vector: {np.array2string(best_model_vec)}")
    res.append(f"Best unbiased model vector: {np.array2string(best_model_unbiased_vec)}")
    res.append(f"Best model index: {best_model_index}")
    res.append(f"Best unbiased model index: {best_model_unbiased_index}")
    export_results(res, "DT", f"dt-{d}_{i}-[{date_time}]")

    # save the archive
    arx.to_pickle(f"{ARCHIVE_DIR}/DT/dt-{d}_{i}-[{date_time}].pkl")



# MULTIPROCESSING
import sys

if __name__ == "__main__":
    import multiprocessing
    processes = []

    EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "BOTH"
    PROCESSES = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    DATASETS = ['synthetic']

    # run test experiment with only 100 iterations
    if "TEST" in EXPERIMENT:
        if "NN" in EXPERIMENT:
            nn_exp('adult', 1, 100)
        if "DT" in EXPERIMENT:
            dt_exp('adult', 1, 100)
        exit(0)

    # nn_exp('adult', 1, EXP_ITERATIONS)
    
    # use MULTIPLE processes
    with multiprocessing.Pool(processes=PROCESSES) as pool:
        for d in DATASETS:
            for i in EXP_RANGE:
                if EXPERIMENT in ["NN", "BOTH"]:
                    print(f"Starting NN experiment {i} on {d} dataset")
                    p = pool.apply_async(nn_exp, args=(d,i,EXP_ITERATIONS))
                    processes.append(p)
                
                if EXPERIMENT in ["DT", "BOTH"]:
                    print(f"Starting DT experiment {i} on {d} dataset")
                    p = pool.apply_async(dt_exp, args=(d,i,EXP_ITERATIONS))
                    processes.append(p)

        # wait for all processes to finish
        for p in processes:
            p.get()
    