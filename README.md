# sketching_optimisation

Examples of random projection sketching methods to reduce the computational
burden of intensive matrix computations.

To do:
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
