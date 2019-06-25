'''
Plots for the LASSO experiments `ihs_timing_real.py`
'''
import itertools
import matplotlib.pyplot as plt
from matplotlib import rc
import json
import os
import sys
sys.path.append('../..')
# from experiment_parameter_grid import param_grid
# from my_plot_styles import plotting_params
# from matplotlib_config import update_rcParams
import numpy as np
from pprint import PrettyPrinter
from data_config import datasets
from plot_config import plotting_params,\
                        my_markers,\
                        my_lines,\
                        update_rcParams,\
                        sketch_names_print_version,\
                        data_names_print_version
import matplotlib.pyplot as plt

update_rcParams()

############################## GLOBALS ##############################
sketches = ['gaussian','srht','countSketch','sjlt']
sampling_factors = [5,10]
#######################################################################


def make_time_plots(data,save_dir, time_results):
    file_suffix = data +  ".pdf"

    # Error to opt plots
    error_file_name = save_dir + "error_time" + file_suffix
    fig, ax = plt.subplots()
    for sketch_method in sketches:
        my_colour = plotting_params[sketch_method]["colour"]
        for gamma in sampling_factors:
            times2test = list(time_results[sketch_method][gamma].keys())
            my_label = sketch_names_print_version[sketch_method] + str(gamma)
            my_line   = my_lines[sampling_factors.index(gamma)]
            my_marker = plotting_params[sketch_method]['marker']
            yvals = [np.log10(time_results[sketch_method][gamma][time]["error to opt"]) for time in times2test]
            ax.plot(times2test, yvals, color=my_colour,linestyle=my_line,
                    marker=my_marker,label=my_label)

    #ax.axvline(sklearn_time,color=plotting_params["Exact"]["colour"],label='Sklearn')
    ax.legend(title='{}'.format(data_names_print_version[data]))
    ax.set_ylabel("$\log (\| \hat{x} - x_{OPT} \|^2_A/n )$", fontsize=18)
    ax.set_xlabel('log(Time(seconds))', fontsize=18)
    #ax.set_xscale('log')
    plt.show()
    #fig.savefig(error_file_name,bbox_inches="tight")

    # Number of iterations plots
    num_iters_file_name = save_dir + "num_iters_time" + file_suffix
    fig, ax = plt.subplots()
    for sketch_method in sketches:
        my_colour = plotting_params[sketch_method]["colour"]
        for gamma in sampling_factors:
            times2test = list(time_results[sketch_method][gamma].keys())
            my_label = sketch_names_print_version[sketch_method] + str(gamma)
            my_line   = my_lines[sampling_factors.index(gamma)]
            my_marker = plotting_params[sketch_method]['marker']
            yvals = [time_results[sketch_method][gamma][time]["num iterations"] for time in times2test]
            ax.plot(times2test, yvals, color=my_colour, linestyle=my_line,
                    marker=my_marker,label=my_label)
    ax.legend(title='{}'.format(data_names_print_version[data]))
    ax.set_ylabel("Number of iterations", fontsize=18)
    ax.set_xlabel('Time (seconds)', fontsize=18)
    #ax.set_xscale('log')
    #fig.savefig(num_iters_file_name,bbox_inches="tight")
    plt.show()




def main():
    directory = "../../figures/ihs_timings/"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Creating directory. ' +  directory)

    # n_trials = time_error_ihs_grid['num trials']
    # for n,d in itertools.product(time_error_ihs_grid['rows'],time_error_ihs_grid['columns']):
    sklearn_lasso_bound = 5.0
    sampling_factors = [5,10]

    for data in datasets.keys():
        if data != 'specular':
            continue
        print("Plotting for {}".format(data))
        print("-"*80)
        file_name = 'ihs_time_' + data + '.npy'
        exp_results = np.load(file_name)[()]
        pretty = PrettyPrinter(indent=4)
        pretty.pprint(exp_results)
        make_time_plots(data,directory,exp_results)

    pass




if __name__ == '__main__':
    main()
