import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch
from copy import deepcopy

from util.arx import get_arx_errors, get_arx_predictions
from util.util import inverse_normalize


def plot_time_series(t, data, labels, ylabel=None, title=None):
    
    if isinstance(data, np.ndarray):
        data = [data]
    plt.figure(figsize=(16,9))
    for i,x in enumerate(data):
        plt.plot(t, x, lw=3, label=labels[i])
    plt.grid()
    if title is not None:
        plt.title(title, size=25)
    plt.xlabel('Time [s]', size=22)
    if ylabel is not None:
        plt.ylabel(ylabel, size=22)
    plt.legend(prop={'size':20})
    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.show()
    
def plot_building_errors(pinode, arx, df, scale=1, retrain=False):
    
    sequences = [seq for seq in pinode.validation_sequences if seq[1] - seq[0] >= 300]
    arx_errors = get_arx_errors(arx, df, sequences, retrain=retrain)
    
    sequences = [x for x in pinode.validation_sequences if x[1] - x[0] >= 300]
    pred, y = pinode.simulate(sequences)
    
    print('\nMean MAE')
    print(f'ARX:\t\t\t{np.mean(arx_errors):.2f}')
    print(f'PiNODE:\t\t\t{np.mean(torch.abs(pred - y).mean(axis=0).mean(axis=1).detach().numpy()):.2f}\n')
    print(f'Improvements')
    print(f'Average:\t\t{100 - (np.mean(torch.abs(pred - y).mean(axis=0).mean(axis=1).detach().numpy()) / np.mean(arx_errors))*100:.1f}%')
    print(f'End of the horizon:\t{100 - (torch.abs(pred - y).mean(axis=0).mean(axis=1).detach().numpy()[-1] / arx_errors[-1])*100:.1f}%\n')

    plt.figure(figsize=(16,7))
    plt.plot(np.arange(len(arx_errors))/4, arx_errors, lw=3*scale, label='ARX', c='tab:orange', ls='--')
    plt.plot(np.arange(len(arx_errors))/4, torch.abs(pred - y).mean(axis=0).mean(axis=1).detach().numpy(), lw=3*scale, label='PC-NODE', c='tab:green', ls='--')
    plt.xlabel('Prediction horizon [h]', size=17*scale)
    plt.ylabel('MAE [$^\circ C$]', size=17*scale)
    plt.xticks(size=17*scale)
    plt.yticks(size=17*scale)
    plt.grid()
    plt.legend(prop={'size':17*scale})
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'building_errors.pdf'), format='pdf')
    plt.show()
    
def plot_building_trajectories(pinode, arx, df_, sequence, scale=1):
    
    data = pinode.X.copy()
    df = df_.copy()
    power = pinode.X[sequence[0]+12:sequence[1], [-5,-1]].copy()
    
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    pred_0, _ = pinode.simulate([sequence])
    pred_0 = pred_0.detach().numpy()
    
    heating = np.mean(power) > 0
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5] if heating else [-3,-2,-1]] = power[:, -2 if heating else -1].copy().reshape(-1,1)
    pred_1, _ = pinode.simulate([sequence])
    pred_1 = pred_1.detach().numpy()
    
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    pinode.X[sequence[0]+12:sequence[1], [-3,-2,-1] if heating else [-7,-6,-5]] = - power[:, -2 if heating else -1].copy().reshape(-1,1)
    pred_2, _ = pinode.simulate([sequence])
    pred_2 = pred_2.detach().numpy()
    
    pinode.X = data.copy()
    
    fig, ax = plt.subplots(4,1,figsize=(16,12), sharey=False, sharex=True)
    for j, p in enumerate([pred_0, pred_1, pred_2]):
        for i in range(3):
            ax[i].plot(pinode.data.index[sequence[0]+12:sequence[1]], p[0,:,i] - 273.15, lw=3*scale, c='black' if j==0 else ('tab:red' if ((heating and j==1) or (not heating and j==2)) else 'tab:blue'))
    ax[-1].plot(pinode.data.index[sequence[0]+12:sequence[1]], [0] * (sequence[1] - sequence[0] - 12), lw=3*scale, label='No power', c='black')
    ax[-1].plot(pinode.data.index[sequence[0]+12:sequence[1]], power.sum(axis=1) / 1000, lw=3*scale, label='Heating' if heating else 'Cooling', c='tab:red' if heating else 'tab:blue')
    ax[-1].plot(pinode.data.index[sequence[0]+12:sequence[1]], - power.sum(axis=1) / 1000, lw=3*scale, label='Heating' if not heating else 'Cooling', c='tab:red' if not heating else 'tab:blue')
    
    for i in range(4):
        ax[i].grid()
        ax[i].set_ylabel(f'$T_{i+1}$ [$^\circ$ C]' if i < 3 else 'Power [kW]', size=17*scale)
        ax[i].tick_params(axis='x', which='major', labelsize=17*scale)
        ax[i].tick_params(axis='y', which='major', labelsize=17*scale)
    for i in range(3):
        ax[i].set_ylim(min([ax[i].get_ylim()[0] for i in range(3)]), max([ax[i].get_ylim()[1] for i in range(3)]))
    ax[-1].legend(prop={'size':17*scale}, bbox_to_anchor=(0.45,-0.4), loc='upper center', ncols=3)
    
    ax[-1].set_xlabel(f'Time', size=17*scale)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))    
    
    fig.tight_layout()
    plt.savefig(os.path.join('plots', 'building_trajectories.pdf'), format='pdf')
    plt.show()
    
def compare_pinode_arx_trajectories(pinode, arx, df_, sequence, scale=1):
    
    data = pinode.X.copy()
    df = df_.copy()
    power = pinode.X[sequence[0]+12:sequence[1], [-5,-1]].copy()
    
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    pred_0, _ = pinode.simulate([sequence])
    pred_0 = pred_0.detach().numpy()
    df.iloc[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    arx_0 = get_arx_predictions(arx, df, sequence)
    
    heating = np.mean(power) > 0
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5] if heating else [-3,-2,-1]] = power[:, -2 if heating else -1].copy().reshape(-1,1)
    pred_1, _ = pinode.simulate([sequence])
    pred_1 = pred_1.detach().numpy()
    df.iloc[sequence[0]+12:sequence[1], [-7,-6,-5] if heating else [-3,-2,-1]] = power[:, -2 if heating else -1].copy().reshape(-1,1)
    arx_1 = get_arx_predictions(arx, df, sequence)
    
    pinode.X[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    df.iloc[sequence[0]+12:sequence[1], [-7,-6,-5,-3,-2,-1]] = 0.
    
    pinode.X[sequence[0]+12:sequence[1], [-3,-2,-1] if heating else [-7,-6,-5]] = - power[:, -2 if heating else -1].copy().reshape(-1,1)
    pred_2, _ = pinode.simulate([sequence])
    pred_2 = pred_2.detach().numpy()
    df.iloc[sequence[0]+12:sequence[1], [-3,-2,-1] if heating else [-7,-6,-5]] = - power[:, -2 if heating else -1].copy().reshape(-1,1)
    arx_2 = get_arx_predictions(arx, df, sequence)
    
    pinode.X = data.copy()
    
    fig, ax = plt.subplots(3, 2,figsize=(32,11), sharey='row', sharex=True)
    for j, p in enumerate([(pred_0, arx_0), (pred_1, arx_1), (pred_2, arx_2)]):
        for i in range(2):
            ax[i,0].plot(pinode.data.index[sequence[0]+12:sequence[1]], p[0][0,:,i] - 273.15, lw=3*scale, c='black' if j==0 else ('tab:red' if ((heating and j==1) or (not heating and j==2)) else 'tab:blue'), label='No power' if j==0 else ('Heating' if ((heating and j==1) or (not heating and j==2)) else 'Cooling'))
            ax[i,1].plot(pinode.data.index[sequence[0]+12:sequence[1]], p[1][0,:,i] - 273.15, lw=3*scale, c='black' if j==0 else ('tab:red' if ((heating and j==1) or (not heating and j==2)) else 'tab:blue'), label='No power' if j==0 else ('Heating' if ((heating and j==1) or (not heating and j==2)) else 'Cooling'))
    for i in range(2):
        ax[-1,i].plot(pinode.data.index[sequence[0]+12:sequence[1]], [0] * (sequence[1] - sequence[0] - 12), lw=3*scale, label='No power', c='black')
        ax[-1,i].plot(pinode.data.index[sequence[0]+12:sequence[1]], power.sum(axis=1) / 1000, lw=3*scale, label='Heating' if heating else 'Cooling', c='tab:red' if heating else 'tab:blue')
        ax[-1,i].plot(pinode.data.index[sequence[0]+12:sequence[1]], - power.sum(axis=1) / 1000, lw=3*scale, label='Heating' if not heating else 'Cooling', c='tab:red' if not heating else 'tab:blue')
    
    for i in range(3):
        for j in range(2):
            ax[i,j].grid()
            ax[i,j].tick_params(axis='x', which='major', labelsize=17*scale)
            ax[i,j].tick_params(axis='y', which='major', labelsize=17*scale)
        ax[i,0].set_ylabel(f'$T_{i+1}$ [$^\circ$C]' if i < 2 else 'Power [kW]', size=17*scale)

    for j in range(2):
        #for i in range(2):
            #ax[i,j].set_ylim(min([ax[i,j].get_ylim()[0] for i in range(2)]), max([ax[i,j].get_ylim()[1] for i in range(2)]))    
        ax[-1,j].set_xlabel(f'Time', size=17*scale)
        ax[-1,j].xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))  

    ax[1,1].legend(prop={'size':17*scale}, bbox_to_anchor=(0.95,0.5), loc='center left', ncols=1)
      
    #fig.autofmt_xdate()
    
    fig.tight_layout()
    plt.savefig(os.path.join('plots', 'compare_pinode_arx_trajectories.pdf'), format='pdf')
    plt.show()
    
def compare_pcnode_arx_trajectories(pinode, arx, df_, sequence, scale=1):
    
    data = pinode.X.copy()
    df = df_.copy()
    
    pred_0, _ = pinode.simulate([sequence])
    pred_0 = pred_0.detach().numpy()
    arx_0 = get_arx_predictions(arx, df, sequence)
    
    fig, ax = plt.subplots(3, 1,figsize=(16,10), sharey=True, sharex=True)
    for i in range(3):
        ax[i].plot(pinode.data.index[sequence[0]+12:sequence[1]], pinode.data.iloc[sequence[0]+12:sequence[1], i+2] - 273.15, lw=3*scale, c='black', label='Measurement')        
        ax[i].plot(pinode.data.index[sequence[0]+12:sequence[1]], arx_0[0,:,i] - 273.15, lw=3*scale, c='tab:orange', ls='--', label='ARX')
        ax[i].plot(pinode.data.index[sequence[0]+12:sequence[1]], pred_0[0,:,i] - 273.15, lw=3*scale, c='tab:green', ls='--', label='PC-NODE')
    
    for i in range(3):
        ax[i].grid()
        ax[i].tick_params(axis='x', which='major', labelsize=17*scale)
        ax[i].tick_params(axis='y', which='major', labelsize=17*scale)
        ax[i].set_ylabel(f'$T_{i+1}$ [$^\circ$C]', size=17*scale)
 
    ax[-1].set_xlabel(f'Time', size=17*scale)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))  

    ax[-1].legend(prop={'size':17*scale}, bbox_to_anchor=(0.45,-0.5), loc='upper center', ncols=3)
      
    #fig.autofmt_xdate()
    
    fig.tight_layout()
    plt.savefig(os.path.join('plots', 'compare_pinode_arx_trajectories.pdf'), format='pdf')
    plt.show()

def plot_piston_spring_trajectories(pcnode, classical_nn, X, X_val, U_val, init_val, min_, max_, scale=1):
    
    pred = pcnode(X_val[:,[0],:], U_val, init_val)
    pred_2 = classical_nn(X_val[:,[0],:], U_val, init_val)
    
    X = inverse_normalize(deepcopy(X), min_, max_)
    X_val = inverse_normalize(deepcopy(X_val), min_, max_)
    pred = inverse_normalize(pred, min_, max_)
    pred_2 = inverse_normalize(pred_2, min_, max_)
    
    pred = pred.detach().numpy()
    pred_2 = pred_2.detach().numpy()
    t = np.linspace(0, pred.shape[1]*pcnode.h, pred.shape[1])
    
    fig, ax = plt.subplots(2,2,figsize=(16,11), sharey=False, sharex='col')
    
    dim = 2
    i = 0
    for i, dim in enumerate([0,2]):
        for j, s in enumerate([3,39]): 
            if i == 1:
                ax[i,j].plot(t, X[s,:, dim]*1000 if i == 0 else X[s,:,dim], lw=2*scale, label='Noisy data' if j==0 else None, c='grey', alpha=0.5)
            ax[i,j].plot(t, X_val[s,:, dim]*1000 if i == 0 else X_val[s,:,dim], lw=2*scale, label='Ground truth' if j==0 else None, c='black')
            ax[i,j].plot(t, pred_2[s,:, dim]*1000 if i == 0 else pred_2[s,:,dim], lw=3*scale, label='NODE' if j==1 else None, c='tab:red', ls='--')
            ax[i,j].plot(t, pred[s,:, dim]*1000 if i == 0 else pred[s,:,dim], lw=3*scale, label='PC-NODE' if j==1 else None, c='tab:green', ls='--')
            ax[i,j].grid()
            ax[i,j].tick_params(axis='x', which='major', labelsize=17*scale)
            ax[i,j].tick_params(axis='y', which='major', labelsize=17*scale)
            ax[1,j].set_xlabel('Time [s]', size=17*scale)
    ax[1,0].legend(prop={'size':17*scale}, bbox_to_anchor=(0.5,-0.35), loc='upper center', ncol=1)
    ax[1,1].legend(prop={'size':17*scale}, bbox_to_anchor=(0.5,-0.35), loc='upper center', ncol=1)
    
    ax[0,0].set_title('Trajectory 1', size=17*scale)
    ax[0,1].set_title('Trajectory 2', size=17*scale)
    ax[0,0].set_ylabel('Entropy [J/K]', size=17*scale)
    ax[1,0].set_ylabel('Position [m]', size=17*scale)
    
    fig.tight_layout()
    plt.savefig(os.path.join('plots', 'piston_trajectories.pdf'), format='pdf')
    plt.show()