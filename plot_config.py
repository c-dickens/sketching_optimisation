'''
file containing the plotting information for matplotlib functions.
Includes line styles, colours etc for each of the sketching methods.
'''
#from experiment_parameter_grid import param_grid
#from experiment_parameter_grid import param_grid
import matplotlib as mpl
mpl.use('TkAgg')
#mpl.use('Agg')
#mpl.use('ps')
import matplotlib.pyplot as plt

sklearn_colour = "C1"
my_markers = ['.', 's', '^', 'D', 'x', '+', 'V', 'o', '*']
#col_markers = {param_grid['columns'][i]: my_markers[i] for i in range(len(param_grid['columns']))}
sketch_names_print_version = {
    'countSketch' : 'CountSketch',
    'sjlt'        : 'SJLT',
    'srht'        : 'SRHT',
    'gaussian'    : 'Gaussian',
    'Sketch & Solve' : 'Sketch \& Solve'
}

data_names_print_version = {
    'w8a' :   'w8a',
    'w6a' :   'w6a',
    'w4a' : 'w4a',
    'covertype' : 'Cover',
    'KDDCup99'  : 'KDD99',
    'aloi' : 'Aloi',
    'APSFailure' : 'APS',
    'albert' : 'Albert',
    'fars' : 'Fars',
    'YearPredictionMSD' : 'Years',
    'abtaha2' : 'Abtaha',
    'specular' : 'Specular'
}

def update_rcParams():
    # This mpl style is from the UCSC BME163 class.
    plt.rcParams.update({
        #'pgf.texsystem'       : 'pdflatex',
        #'backend'             : 'ps',
        'font.size'           : 12.0      ,
        'font.family'         : 'DejaVu Sans',
        'xtick.major.size'    : 4        ,
        'xtick.major.width'   : 0.75     ,
        'xtick.labelsize'     : 12.0      ,
        'xtick.direction'     : 'out'      ,
        'ytick.major.size'    : 4        ,
        'ytick.major.width'   : 0.75     ,
        'ytick.labelsize'     : 12.0      ,
        'ytick.direction'     : 'out'      ,
        'xtick.major.pad'     : 2        ,
        'xtick.minor.pad'     : 2        ,
        'ytick.major.pad'     : 2        ,
        'ytick.minor.pad'     : 2        ,
        'savefig.dpi'         : 900      ,
        'axes.linewidth'      : 0.75     ,
        'text.usetex'         : True     ,
        'text.latex.unicode'  : False     })

plotting_params = {"countSketch" : {"colour" : "b",
                                    "line_style" : '-',
                                    "marker" : "o" },
                  "sjlt" : {"colour" : "m",
                                 "line_style" : '-',
                                 "dashes" : [8, 4, 2, 4, 2, 4], #mpl only has 4 linestyles so do custom dashes
                                 "marker" : "d" },
                   "srht" : {"colour" : "k",
                             "marker" : "s",
                             "line_style" : ':'},
                   "gaussian" : {"colour" : "r",
                                 "marker" : "v",
                                 "line_style" : "-."},
                   "Sketch & Solve" : {"colour" : "g",
                                  "marker" : "x",
                                  "line_style" : '-',
                                  "dashes" : [2, 4, 2, 2, 4, 2]
                                  }, #mpl only has 4 linestyles so do custom dashes},
                    "Exact" : {"colour" : "cyan",
                               "marker" : "*",
                                "line_style" : ':'}
                                  }

# nb. the marker styles are for the plots with multiple sketch settings.
my_markers = ['.', 's', '^', 'D', 'x', '+', 'V', 'o', '*', 'H']
my_lines   = ['-', ':', '--','-.']
#col_markers = {param_grid['columns'][i]: my_markers[i] for i in range(len(param_grid['columns']))}
#print(col_markers)
