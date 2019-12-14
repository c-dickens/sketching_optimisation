# sketching_optimisation

Examples of random projection sketching methods to reduce the computational
burden of intensive matrix computations.

All datasets can be downloaded in the required format from https://drive.google.com/open?id=1w-EwPNmi-qiddui1RfSBhbrylL_8LlgN
Note that these must be placed in the `data/` directory e.g. `data/YearPredictionMSD.npy` 

### Completed Experiments:
Experiment `exp` is located in `experiments/` and the corresponding output is found in
`output/exp`.
Note that there will be intermediate directories in the above substitution.
1. `baselines/metadata.py` -- Computes the basic metadata for the real world datasets used.
2. `baselines/summary_performance.py` -- Performs a comparison (time and error) of the competing methods.

To run these experiments:
0. Install external dependency for Fast Hadamard Transform (see below)
1. Ensure that the necessary datasets are downloaded.  The UCI ones have the url hardcoded,
howevever you will need to follow the urls in the `all_datasets` dictionary in `get_datasets.py`
to download the libsvm files and the suite-sparse datasets.
These *must* be saved in the same directory as `get_datasets.py`.
The script will automatically download the UCI and OPENML datasets.
Open-ml datasets will be downloaded automatically by running the script.
This file must be compiled and run from the directory (I don't know why)

### To do:

0. Running the profile script it is clear that the bottleneck is repeated
conversion of the `ndarray` data type to the `coo_matrix`.
Two things can be done:
(i) Either allow the functions to better accept sparse matrix as input
(ii) Convert the data and add sparse references (`row`, `col` etc)
If better handling of sparse vs dense data can be done in the `rp.__init__`
method then this would allow for random number generation within the `rp.sketch`
method which would be better for repeated calls in IHS.
1. Refactor and test the solvers for IHS versions of OLS,RIDGE & LASSO
2. Start refactoring the subspace embedding experiments
3. Generate data metadata scripts and plots

## Installation of fastwht repo:
<!-- 1. `git clone` the repo
2. `cd matrix_sketching`
3. `pip install -r requirements.txt`
4. `cd matrix_sketching/lib` -->
1. `git clone https://bitbucket.org/vegarant/fastwht.git` --> then run install
in here by `cd python`, `python setup.py`, `python test.py`
1. Get the directory path for `fastwht/python` which should be `your_path_name =
*/sketching_optimisation/lib/fastwht/python`
3. Find `.bash_profile` or equivalent and add `export PYTHONPATH=$PYTHONPATH:your_path_name`,
at the final line of the `bash_profile`, finally save then `source .bash_profile`.
4. Open ipython, do `import sys --> sys.path` and check that `your_path_name`
is displayed.
5. Go back to the `sketching_optimisation` directory and run the tests.

### Extensions
- Doing this with more general functions akin to Newton Sketch
- Bias-variance tradeoff with this estimation method?
- Combine this with SGD?
- Rank-deficient of weak embeddings? Is there a range of results here?
- SVM or SVR?
