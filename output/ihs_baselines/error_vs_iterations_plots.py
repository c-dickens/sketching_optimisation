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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
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


##### EXPERIMENTAL GLOBAL PARAMETERS
update_rcParams()
np.set_printoptions(3)
SAVE_DIR = '/figures/baselines/'
CURRENT_DIR = os.getcwd()
def plot_error_iterations_opt():
    '''Generate the plots for the IHS OLS baseline experiments while measuring
    the error to x_opt and x* over a sequence of iterations'''

    error2opt = CURRENT_DIR + '/error_vs_iters_opt_5_10.npy'



    opt_error = np.load(error2opt)[()]
    print(opt_error)

    # opt_fig, opt_ax = plt.subplots()
    # for sketch in opt_error.keys():
    #     print('*'*80)
    #     print('\n')
    #     for gamma in list(opt_error[sketch].keys()):
    #         gamma_id = list(opt_error[sketch].keys()).index(gamma)
    #         errors = opt_error[sketch][gamma]
    #         print(errors)
    #         iters = [_+1 for _ in range(len(errors))]
    #
    #         my_label =  sketch_names_print_version[sketch] + str(gamma)
    #         my_colour  = plotting_params[sketch]["colour"]
    #         my_marker =  plotting_params[sketch]["marker"]
    #         my_line = my_lines[gamma_id]
    #         opt_ax.plot(iters,errors,
    #                     color=my_colour,
    #                     marker=my_marker,
    #                     linewidth=2,
    #                     markersize=6,
    #                     linestyle=my_line,
    #                     label=my_label)
    opt_fig, opt_ax, opt_ticks = plot_exp_data(opt_error)
    opt_ax.set_yscale('log')
    opt_ax.legend()
    opt_ax.set_xlabel('Iterations')
    opt_ax.set_ylabel('Prediction Error: $\|\hat{x} - x_{OPT}\|_A^2$')
    #ticks = [_ for _ in iters if _ % 5 ==0]
    #ticks.insert(0,1)
    opt_ax.set_xticks(opt_ticks)
    #plt.show()
    plt.tight_layout()
    current_dir = os.getcwd()
    print(current_dir)
    save_dir = '../../figures/ihs_baselines/'
    fname = save_dir + 'ols_error2opt_5_10.pdf'
    opt_fig.savefig(fname)


def plot_error_iterations_truth():
    '''Could use a generic function but we need more control
    for the zoomed part of the plot.
    Might need to focus in on one bit which shows plateau at
    the 0.034 value.

    NB. The results here plateau out at 0.034 which is roughly
    the square of 0.2.  This is why the absolute values are slightly
    different to thos in the original IHS paper as they work with
    the squre root of out result.
    '''
    error2truth = CURRENT_DIR + '/error_vs_iters_truth_5_10.npy'



    truth_error = np.load(error2truth)[()]
    print(truth_error)

    truth_fig, truth_ax = plt.subplots()
    for sketch in truth_error.keys():
        print('*'*80)
        print('\n')
        for gamma in list(truth_error[sketch].keys()):
            gamma_id = list(truth_error[sketch].keys()).index(gamma)
            errors = truth_error[sketch][gamma]
            print(errors)
            iters = [_+1 for _ in range(len(errors))]

            my_label =  sketch_names_print_version[sketch] + str(gamma)
            my_colour  = plotting_params[sketch]["colour"]
            my_marker =  plotting_params[sketch]["marker"]
            my_line = my_lines[gamma_id]
            truth_ax.plot(iters,errors,
                        color=my_colour,
                        marker=my_marker,
                        linewidth=2,
                        markersize=6,
                        linestyle=my_line,
                        label=my_label)
    truth_ax.legend()
    truth_ax.set_xlabel('Iterations')
    truth_ax.set_ylabel('Prediction Error: $\|\hat{x} - x^*\|_A^2$')
    truth_ticks = [_ for _ in iters if _ % 5 ==0]
    truth_ticks.insert(0,1)
    truth_ax.set_xticks(truth_ticks)
    #plt.show()
    plt.tight_layout()
    current_dir = os.getcwd()
    print(current_dir)
    save_dir = '../../figures/ihs_baselines/'
    fname = save_dir + 'ols_error2truth_5_10.pdf'
    truth_fig.savefig(fname)

def plot_error_iterations_truth_zoomed():
    '''Could use a generic function but we need more control
    for the zoomed part of the plot.
    Might need to focus in on one bit which shows plateau at
    the 0.034 value.

    NB. The results here plateau out at 0.034 which is roughly
    the square of 0.2.  This is why the absolute values are slightly
    different to thos in the original IHS paper as they work with
    the squre root of out result.

    Adapted from: http://akuederle.com/matplotlib-zoomed-up-inset
    '''
    error2truth = CURRENT_DIR + '/error_vs_iters_truth.npy'



    truth_error = np.load(error2truth)[()]
    print(truth_error)

    truth_fig, truth_ax = plt.subplots()
    for sketch in truth_error.keys():
        print('*'*80)
        print('\n')
        for gamma in list(truth_error[sketch].keys()):
            if gamma < 6:
                continue
            gamma_id = list(truth_error[sketch].keys()).index(gamma)
            errors = truth_error[sketch][gamma]
            print(errors)
            iters = [_+1 for _ in range(len(errors))]

            my_label =  sketch_names_print_version[sketch] + str(gamma)
            my_colour  = plotting_params[sketch]["colour"]
            my_marker =  plotting_params[sketch]["marker"]
            my_line = my_lines[gamma_id]
            truth_ax.plot(iters,errors,
                        color=my_colour,
                        marker=my_marker,
                        linewidth=2,
                        markersize=6,
                        linestyle=my_line,
                        label=my_label)

            # zoom-factor: 2.5, location: upper-left
            print(f'Plotting {sketch}, {gamma}')
            axins = zoomed_inset_axes(truth_ax, 2.5, loc=1)
            axins.plot(iters, errors,
                       color=my_colour,
                       marker=my_marker,
                       linewidth=2,
                       markersize=6,
                       linestyle=my_line,
                       label=my_label)
            x1, x2, y1, y2 = 1, 6, 0, 0.09 # specify the limits
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            axin_ticks = [0.0,0.03,0.06,0.09]
            mark_inset(truth_ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            axins.set_yticks(axin_ticks)

    axins.plot(range(x1,x2+1),0.034*np.ones(len(range(x1,x2+1))))


    #truth_ax.legend()
    truth_ax.set_xlabel('Iterations')
    truth_ax.set_ylabel('$\|\hat{x} - x^*\|_A^2$')
    truth_ticks = [_ for _ in iters if _ % 5 ==0]
    truth_ticks.insert(0,1)
    truth_ax.set_xticks(truth_ticks)
    plt.show()


def plot_exp_data(_dict):
    _fig, _ax = plt.subplots()
    for sketch in _dict.keys():
        print('*'*80)
        print('\n')
        for gamma in list(_dict[sketch].keys()):
            gamma_id = list(_dict[sketch].keys()).index(gamma)
            errors = _dict[sketch][gamma]
            print(errors)
            iters = [_+1 for _ in range(len(errors))]

            my_label =  sketch_names_print_version[sketch] + str(gamma)
            my_colour  = plotting_params[sketch]["colour"]
            my_marker =  plotting_params[sketch]["marker"]
            my_line = my_lines[gamma_id]
            _ax.plot(iters,errors,
                        color=my_colour,
                        marker=my_marker,
                        linewidth=2,
                        markersize=6,
                        linestyle=my_line,
                        label=my_label)
    _ticks = [_ for _ in iters if _ % 5 ==0]
    _ticks.insert(0,1)
    return _fig, _ax, _ticks


def main():
    plot_error_iterations_opt()
    plot_error_iterations_truth()
    #plot_error_iterations_truth_zoomed()

if __name__ == '__main__':
    main()
