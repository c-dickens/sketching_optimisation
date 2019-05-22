import json
from pprint import PrettyPrinter
import numpy as np
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix, csr_matrix
import sys
sys.path.append('../..')
from data_config import datasets

def data_metadata():
    '''For a given dataset evaluate:
    - density
    - shape
    - rank
    '''

    # Write to metadata dict
    metadata = {}


    for data in datasets.keys():
        print("-"*80)
        print("Dataset: {}".format(data))
        input_file = datasets[data]["filepath"]
        input_dest = datasets[data]["input_destination"]
        if datasets[data]['sparse_format'] == True:
            df = load_npz('../../' + input_file)
            df = df.tocsr()
        else:
            df = np.load('../../' + input_file)

        X = df[:,:-1]
        n,d = X.shape
        aspect_ratio = d/n

        if isinstance(X,csr_matrix):
            print('Sparse operations')
            nnz = X.getnnz()
            density = nnz / (n*d)
            # nb in here we are really using a low rank approx for the
            # leverage scores as we are projecting onto NUM_SING_VECTORS
            # Can grow up to d but takes longer to compute the scores.
            NUM_VECTORS_FOR_PROJECTION = np.int(d)-1
            U, _,_ = sparse.linalg.svds(X,NUM_VECTORS_FOR_PROJECTION)
            lev_scores = np.linalg.norm(U, axis=1)**2
            coherence = np.max(lev_scores)
            coherence_ratio = coherence / np.min(lev_scores)
            rank = d
        else:
            print('Dense operations')
            aspect_ratio = d/n
            nnz = np.count_nonzero(X)
            density = nnz/(n*d)
            q,_ = np.linalg.qr(X)
            print(q.shape)
            print(q[0,:10])
            lev_scores = np.linalg.norm(q, axis=1)**2
            print(lev_scores)
            # U,sings,_ = np.linalg.svd(X)
            # lev_scores = np.linalg.norm(U, axis=1)**2
            print('lev scores done')
            coherence = np.max(lev_scores)
            coherence_ratio = coherence / np.min(lev_scores)
            #rank = np.linalg.matrix_rank(X)

        print("Shape: {}".format((n,d)))
        print("Aspect ratio : {}".format(aspect_ratio))
        print("Density: {} ".format(density))
        print("Lev score sum: {}".format(np.sum(lev_scores)))
        print("Coherence {}".format(coherence))
        print("Coherence ratio: {} ".format(coherence / np.min(lev_scores[lev_scores > 0])))

        metadata[data] = {
                    "shape"           : (n,d),
                    "aspect ratio"    : aspect_ratio,
                    "density"         : density,
                    "rank"            : np.int(round(np.sum(lev_scores))),
                    "coherence"       : coherence,
                    "coherence_ratio" : coherence / np.min(lev_scores[lev_scores > 0])
        }
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(metadata)
    with open('../../output/baselines/data_metadata.json', 'w') as outfile:
       json.dump(metadata, outfile)

if __name__ == "__main__":
    data_metadata()
