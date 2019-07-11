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
bar_patterns = ('-', 'o','+','*','O', '\\',  'x',)
SAVE_DIR = '../../figures/ihs_baselines/'
CURRENT_DIR = os.getcwd()

def plot_error_vs_dims():
    '''Generate the bar plots for the IHS OLS baseline experiments while measuring
    the error to x* whil varying d'''

    error2truth_path = CURRENT_DIR + '/error_vs_dims.npy'
    error2truth = np.load(error2truth_path)[()]
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(error2truth)

    dimensions_tested = error2truth['Dimensions']
    dim_labels = {16: {
                    'Exact' : 'Exact',
                    'Sketch & Solve' : 'Sketch \& Solve',
                    'countSketch' : 'CountSketch',
                    'gaussian' : 'Gaussian',
                    'sjlt' : 'SJLT',
                    'srht' : 'SRHT'
                    }
                }
    for d in dimensions_tested:
        if d == 16:
            continue
        else:
            dim_labels[d] = {
                'Exact' : None,
                'Sketch & Solve' : None,
                'countSketch' : None,
                'gaussian' : None,
                'sjlt' : None,
                'srht' : None
            }
    index = range(len(dimensions_tested))

    bar_width = 0.15
    fig, ax = plt.subplots()

    for ii in index:
        d = dimensions_tested[ii]


        exact_rects = ax.bar(index[ii],
                            error2truth["Exact"][d],
                            bar_width,
                            color=plotting_params['Exact']["colour"],
                            label=dim_labels[d]['Exact'])
        for bar in exact_rects:
            bar.set_hatch(2*bar_patterns[0])

        classical_rects = ax.bar(index[ii]+bar_width,
                                 error2truth["Sketch & Solve"][d],
                                  bar_width,
                                  color=plotting_params['Sketch & Solve']["colour"],
                                  label=dim_labels[d]['Sketch & Solve'])
        for bar in classical_rects:
            bar.set_hatch(2*bar_patterns[1])

        countsketch_rects=classical_rects = ax.bar(index[ii]+2*bar_width,
                                                error2truth["countSketch"][d],
                                                bar_width,
                                                color=plotting_params['countSketch']["colour"],
                                                label=dim_labels[d]['countSketch'])
        for bar in countsketch_rects:
            bar.set_hatch(2*bar_patterns[2])

        sjlt_rects=classical_rects = ax.bar(index[ii]+3*bar_width,
                                                error2truth["sjlt"][d],
                                                bar_width,
                                                color=plotting_params['sjlt']["colour"],
                                                label=dim_labels[d]['sjlt'])
        for bar in sjlt_rects:
            bar.set_hatch(2*bar_patterns[3])



        srht_rects=classical_rects = ax.bar(index[ii]+4*bar_width,
                                    error2truth["srht"][d],
                                    bar_width,
                                    color=plotting_params['srht']["colour"],
                                    label=dim_labels[d]['srht'])
        for bar in srht_rects:
            bar.set_hatch(2*bar_patterns[4])

        gaussian_rects=classical_rects = ax.bar(index[ii]+5*bar_width,
                                                error2truth["gaussian"][d],
                                                bar_width,
                                                color=plotting_params['gaussian']["colour"],
                                                label=dim_labels[d]['gaussian'])
        for bar in gaussian_rects:
            bar.set_hatch(2*bar_patterns[5])
    # rects = [exact_rects,classical_rects,countsketch_rects,\
    #         sjlt_rects,srht_rects,gaussian_rects]
    # for bar in rects:
    #     bar.set_hatch(2*bar_patterns[rects.index(bar)])

    ax.legend()
    ax.set_xticks(np.asarray(index,dtype=np.float) + 2*bar_width)
    ax.set_xticklabels(dimensions_tested)
    ax.set_ylabel('$\|\hat{x} - x^*\|_A^2$')
    ax.set_xlabel('Dimension $d$')
    #plt.show()
    plt.tight_layout()

    fname = SAVE_DIR + 'ols_error_vs_d.pdf'
    fig.savefig(fname)


def main():
    plot_error_vs_dims()

if __name__ == '__main__':
    main()
