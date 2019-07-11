import json
import itertools
from timeit import default_timer
from pprint import PrettyPrinter
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix, csr_matrix
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../..')
from data_config import datasets
from plot_config import plotting_params,\
                        my_markers,\
                        my_lines,\
                        update_rcParams,\
                        sketch_names_print_version,\
                        data_names_print_version
from experiments.ihs_baselines.error_vs_row_dim import ROWDIMS
##### EXPERIMENTAL GLOBAL PARAMETERS
update_rcParams()
np.set_printoptions(3)
#ROWDIMS = [100*2**i for i in range(3,14)]
# bar_patterns = ('-', 'o','+','*','\\', 'O', 'x',)
# bar_cols = ['lightcoral','bisque','lightsteelblue', 'thistle','white']
# data2time = ['w8a', 'w6a']
###################################

def plot_ihs_ols_error2opt():
    '''Generate the plots for the IHS OLS baseline experiments while measuring
    the error to x_opt'''
    current_dir = os.getcwd()
    mse_exp_results = current_dir + '/ihs_ols_mse_OPT.npy'
    pe_exp_results = current_dir + '/ihs_ols_pred_error_OPT.npy'
    save_dir = '/figures/baselines/'
    print(mse_exp_results)

    MSE = np.load(mse_exp_results)[()]
    PRED_ERROR = np.load(pe_exp_results)[()]
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(MSE)
    pretty.pprint(PRED_ERROR)
    fig, (ax0, ax1) = plt.subplots(1,2)

    for sketch_method in MSE.keys():
        my_colour = plotting_params[sketch_method]["colour"]
        my_marker = plotting_params[sketch_method]["marker"]
        my_line = plotting_params[sketch_method]["line_style"]
        if sketch_method is "Exact":
            my_label = "Optimal"
        else:
            try:
                my_label = sketch_names_print_version[sketch_method]
            except:
                my_label = sketch_method
        ax0.plot(ROWDIMS, MSE[sketch_method],
                 color=my_colour, marker=my_marker,
                 linewidth=2, linestyle=my_line, markersize=6, label=my_label)
        ax1.plot(ROWDIMS, PRED_ERROR[sketch_method],
                color=my_colour, marker=my_marker,
                linewidth=2, linestyle=my_line, markersize=6, label=my_label)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel("Row dimension $n$")
    ax0.set_ylabel("$\|\hat{x} - x_{opt}\|_2^2$")
    ax0.legend()
    ax0.grid(True)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.set_xlabel("Row dimension $n$")
    ax1.set_ylabel("$\|\hat{x} - x_{opt}\|_A^2$")
    ax1.legend()
    plt.tight_layout()
    plt.show()

def plot_ihs_ols_error2truth():
    '''Generate the plots for the IHS OLS baseline experiments while measuring
    the error to x_opt'''
    current_dir = os.getcwd()
    mse_exp_results = current_dir + '/ihs_ols_mse_TRUTH.npy'
    pe_exp_results = current_dir + '/ihs_ols_pred_error_TRUTH.npy'
    save_dir = '/figures/baselines/'
    print(mse_exp_results)

    MSE = np.load(mse_exp_results)[()]
    PRED_ERROR = np.load(pe_exp_results)[()]
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(MSE)
    pretty.pprint(PRED_ERROR)
    fig, (ax0, ax1) = plt.subplots(1,2)

    # Cheap hack just to plot the exact method first and the sketches on top
    methods = list(MSE.keys())
    for sketch_method in methods[::-1]:
        my_colour = plotting_params[sketch_method]["colour"]
        my_marker = plotting_params[sketch_method]["marker"]
        my_line = plotting_params[sketch_method]["line_style"]
        if sketch_method == "Exact":
            my_label = "Optimal"
            my_size = 18
        else:
            my_size = 6
            try:
                my_label = sketch_names_print_version[sketch_method]
            except:
                my_label = sketch_method

        ax0.plot(ROWDIMS, MSE[sketch_method],
                 color=my_colour, marker=my_marker,
                 linewidth=2, linestyle=my_line,
                 markersize=my_size, label=my_label)
        ax1.plot(ROWDIMS, PRED_ERROR[sketch_method],
                color=my_colour, marker=my_marker,
                linewidth=2, linestyle=my_line,
                markersize=my_size, label=my_label)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel("Row dimension $n$")
    ax0.set_ylabel("Solution Error: $\|\hat{x} - x^*\|_2^2$")
    ax0.legend()
    ax0.grid(True)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.set_xlabel("Row dimension $n$")
    ax1.set_ylabel("Prediction Error: $\|\hat{x} - x^*\|_A^2$")
    #ax1.legend()
    plt.tight_layout()
    #plt.show()

    current_dir = os.getcwd()
    print(current_dir)
    save_dir = '../../figures/ihs_baselines/'
    fname = save_dir + 'ols_solution_recovery.pdf'
    fig.savefig(fname)

def main():
    #plot_ihs_ols_error2opt()
    plot_ihs_ols_error2truth()

if __name__ == '__main__':
    main()
