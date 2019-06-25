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

##### EXPERIMENTAL GLOBAL PARAMETERS
update_rcParams()
np.set_printoptions(3)
bar_patterns = ('-', 'o','+','*','\\', 'O', 'x',)
bar_cols = ['lightcoral','bisque','lightsteelblue', 'thistle','white']
data2time = ['w8a', 'w6a']
SAVE_DIR = "../../figures/baselines/"
###################################

def plot_summary_times():
    '''Generate the plots for the summary time experiments'''
    current_dir = os.getcwd()
    exp_results = current_dir + '/summary_time_quality_results.json'
    save_dir = '/figures/baselines/'
    print(exp_results)

    with open(exp_results) as json_file:
        results = json.load(json_file)



    for dataset in results.keys():
        print('*'*80)
        print(dataset)

        dataset_results = results[dataset]
        sketches = dataset_results.keys()



        # Time plots
        time_fname = save_dir + '/sketch_time/_time_sketch_dim_' + dataset +'.pdf'
        time_fig, time_ax = plt.subplots()
        time_fig.set_figheight(3)
        bar_width = 0.1
        bar_index = np.arange(len(sketches)) # need this as an array later on.
        time_bar_groups = len(sketches)

        time_ax.set_ylabel('Sketching Time')
        # use the same set of projection dims for all sketches
        # use gaussian as the dummy variable to pull this out.
        projection_dims = list(dataset_results['gaussian'].keys())
        sorted_random_projections = ['countSketch', 'sjlt','srht','gaussian']
        time2plot = np.zeros((len(projection_dims),len(sketches)))
        print(time2plot)

        for sketch in sorted_random_projections:
            # write the experimental data to an array which can be used to plot.
            # for gamma in dataset_results[sketch].keys():
            #     sketching_times = dataset_results[sketch][gamma]

            sketch_idx = sorted_random_projections.index(sketch)
            #print(sketch,sketch_idx) # this is to see indexing of the cols
            sketch_time = [dataset_results[sketch][γ]['sketch_time'] for γ in projection_dims]
            time2plot[:,sketch_idx] = sketch_time


        int_proj_dims = np.array([np.int(_) for _ in projection_dims])
        int_proj_dims = int_proj_dims[:,None]
        cols = ['gamma'] + sorted_random_projections
        time_df_array = np.c_[int_proj_dims,time2plot]
        print(time_df_array)
        time_df = pd.DataFrame(time_df_array,columns=cols)
        time_df = time_df.set_index('gamma')
        print(time_df)

        time_ax.set_xticks(bar_index + len(projection_dims)*(bar_width/ 2))
        for tick in time_ax.get_xticklabels():
            tick.set_rotation(45)
        time_ax.set_xticklabels([sketch_names_print_version[sketch] for sketch in sorted_random_projections])

        for row_id in range(len(projection_dims)):
            print(row_id)
            x_locations = bar_index + row_id*bar_width
            my_label = r'$\gamma = $' + str(projection_dims[row_id])
            time_rects = time_ax.bar(x_locations, time2plot[row_id],bar_width,
                                color=bar_cols[row_id],edgecolor='black',
                                label=my_label)
            for bar in time_rects:
                bar.set_hatch(2*bar_patterns[row_id])

        # Time plots - reducing size
        time_ax.set_yscale('log')
        time_ax.set_ylim(bottom=1E-5)
        time_ax.legend(loc=2,ncol=2,frameon=False,
                        title=data_names_print_version[dataset])
        plt.show()

    pretty = PrettyPrinter(indent=4)
    pretty.pprint(results)

def plot_summary_errors():
    '''Generate the plots for the summary error experiments.

    The error is empirical frobenius norm error.'''
    current_dir = os.getcwd()
    exp_results = current_dir + '/summary_time_quality_results.json'
    save_dir = '/figures/baselines/'
    print(exp_results)

    with open(exp_results) as json_file:
        results = json.load(json_file)



    for dataset in results.keys():
        print('*'*80)
        print(dataset)


        dataset_results = results[dataset]
        sketches = dataset_results.keys()



        # error plots
        error_fname = save_dir + '/sketch_error/_error_sketch_dim_' + dataset +'.pdf'
        error_fig, error_ax = plt.subplots()
        error_fig.set_figheight(3)
        bar_width = 0.1
        bar_index = np.arange(len(sketches)) # need this as an array later on.
        error_bar_groups = len(sketches)

        error_ax.set_ylabel('Sketching Error')
        # use the same set of projection dims for all sketches
        # use gaussian as the dummy variable to pull this out.
        projection_dims = list(dataset_results['gaussian'].keys())
        sorted_random_projections = ['countSketch', 'sjlt','srht','gaussian']
        error2plot = np.zeros((len(projection_dims),len(sketches)))
        print(error2plot)

        for sketch in sorted_random_projections:
            # write the experimental data to an array which can be used to plot.
            # for gamma in dataset_results[sketch].keys():
            #     sketching_errors = dataset_results[sketch][gamma]

            sketch_idx = sorted_random_projections.index(sketch)
            #print(sketch,sketch_idx) # this is to see indexing of the cols
            sketch_error = [dataset_results[sketch][γ]['frob_error'] for γ in projection_dims]
            error2plot[:,sketch_idx] = sketch_error


        int_proj_dims = np.array([np.int(_) for _ in projection_dims])
        int_proj_dims = int_proj_dims[:,None]
        cols = ['gamma'] + sorted_random_projections
        error_df_array = np.c_[int_proj_dims,error2plot]
        print(error_df_array)
        error_df = pd.DataFrame(error_df_array,columns=cols)
        error_df = error_df.set_index('gamma')
        print(error_df)

        error_ax.set_xticks(bar_index + len(projection_dims)*(bar_width/ 2))
        for tick in error_ax.get_xticklabels():
            tick.set_rotation(45)
        error_ax.set_xticklabels([sketch_names_print_version[sketch] for sketch in sorted_random_projections])

        for row_id in range(len(projection_dims)):
            print(row_id)
            x_locations = bar_index + row_id*bar_width
            my_label = r'$\gamma = $' + str(projection_dims[row_id])
            error_rects = error_ax.bar(x_locations, error2plot[row_id],bar_width,
                                color=bar_cols[row_id],edgecolor='black',
                                label=my_label)
            for bar in error_rects:
                bar.set_hatch(2*bar_patterns[row_id])

        # error plots - reducing size
        error_ax.set_yscale('log')
        error_ax.set_ylim(bottom=1E-2)
        error_ax.legend(loc=2,ncol=2,frameon=False,
                        title=data_names_print_version[dataset])
        plt.show()

def plot_speedups():
    '''Generate the plots for the summary speedups.


    '''
    current_dir = os.getcwd()
    exp_results = current_dir + '/summary_time_quality_results.json'
    save_dir = '/figures/baselines/'
    print(exp_results)
    plotting_dict = {}

    with open(exp_results) as json_file:
        results = json.load(json_file)


    # Time plots
    speedup_fig, speedup_ax = plt.subplots()
    speedup_ax.grid()
    for dataset in results.keys():
        # if dataset not in data2time:
        #     continue
        print('*'*80)
        print(dataset)

        dataset_results = results[dataset]
        sketches = dataset_results.keys()




        speedup_fig.set_figheight(3)
        bar_width = 0.1
        bar_index = np.arange(len(sketches)) # need this as an array later on.
        speedup_bar_groups = len(sketches)

        speedup_ax.set_ylabel('Speedup')
        # use the same set of projection dims for all sketches
        # use gaussian as the dummy variable to pull this out.
        projection_dims = list(dataset_results['gaussian'].keys())
        sorted_random_projections = ['countSketch', 'sjlt','srht','gaussian']
        time2plot = np.zeros((len(projection_dims),len(sketches)))
        print(time2plot)

        for sketch in sorted_random_projections:
            # write the experimental data to an array which can be used to plot.
            # for gamma in dataset_results[sketch].keys():
            #     sketching_times = dataset_results[sketch][gamma]

            sketch_idx = sorted_random_projections.index(sketch)
            #print(sketch,sketch_idx) # this is to see indexing of the cols
            sketch_time = [dataset_results[sketch][γ]['sketch_time'] for γ in projection_dims]
            time2plot[:,sketch_idx] = sketch_time


        int_proj_dims = np.array([np.int(_) for _ in projection_dims])
        int_proj_dims = int_proj_dims[:,None]
        cols = ['gamma', 'speedup']
        speedup_df_array = np.c_[int_proj_dims,time2plot]
        speedup_df_array = np.c_[speedup_df_array[:,0], speedup_df_array[:,3]/speedup_df_array[:,1]]
        print(speedup_df_array)
        mean_speedup = np.mean(speedup_df_array[:,1],axis=0)
        plotting_dict[dataset] = mean_speedup
        speedup_df = pd.DataFrame(speedup_df_array,columns=cols)
        speedup_df = speedup_df.set_index('gamma')
        print(speedup_df)




        speedup_ax.set_xticks(bar_index + len(projection_dims)*(bar_width/ 2))
        for tick in speedup_ax.get_xticklabels():
            tick.set_rotation(45)
        speedup_ax.set_xticklabels(list(plotting_dict.keys()))

        # for row_id in range(len(projection_dims)):
        #     print(row_id)
        #     x_locations = bar_index + row_id*bar_width
        #     my_label = r'$\gamma = $' + str(projection_dims[row_id])
        #     speedup_rects = speedup_ax.bar(x_locations, time2plot[2]/time2plot[0],bar_width,
        #                         color=bar_cols[row_id],edgecolor='black',
        #                         label=my_label)
        #     for bar in speedup_rects:
        #         bar.set_hatch(2*bar_patterns[row_id])

        # Time plots - reducing size
        #speedup_ax.set_yscale('log')
        #speedup_ax.set_ylim(bottom=1E-5)
        #speedup_ax.legend(loc=2,ncol=2,frameon=False)
        #plt.show()

    # # pretty = PrettyPrinter(indent=4)
    # # pretty.pprint(results)
    print(plotting_dict)
    plotting_df = pd.DataFrame.from_dict(plotting_dict,orient='index')
    plotting_df.sort_values(by=0,inplace=True,ascending=False)
    print(plotting_df)
    plotting_df.plot.bar(sort_columns=True,
                         legend=False,ax=speedup_ax)
    dataset_labels = [data_names_print_version[_] for _ in plotting_df.index]
    speedup_ax.set_xticklabels(dataset_labels,rotation=45)
    speedup_ax.set_ylabel('Summary Speedup')
    #speedup_ax.bar(range(len(plotting_dict)),plotting_dict.values(),align='center')
    # speedup_ax.set_xticks(range(len(plotting_dict)), list(plotting_dict.keys()))
    #speedup_ax.set_yscale('log')
    #speedup_ax.set_xticklabels(list(plotting_dict.keys()))
    plt.show()
    current_dir = os.getcwd()
    save_loc = SAVE_DIR+'speedups.pdf'
    print('Saving at ',save_loc)
    speedup_fig.savefig(save_loc,bbox_inches="tight")


if __name__ == "__main__":
    #plot_summary_times()
    #plot_summary_errors()
    plot_speedups()
