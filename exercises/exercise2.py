#---------------  EXERCISE 2 ---------------#

"""
Exercise 2. Construct a web consisting of three or more subwebs
and verify that dim(V1(A)) equals (or exceeds) the number of
the components in the web.
"""

"""
Let's define component: 
Component of a web = a group of pages that are connected with 
each other (directed or undirected), but not connected to the rest of the web.

And V1(A):
Eigenspace of the matrix A

dim(V1(A)):
tells us the amount of subwebs. In other words, how many independent pagerank
spaces exist in the web. of course we only take the ones with eigval = 1
"""

import numpy as np
from src import create_link_matrix, is_column_stochastic

if __name__ == "__main__":
    links = {
        1: [2, 3],
        2: [1, 3],
        3: [2],
        4: [5, 6],
        5: [4, 6],
        6: [5, 4],
        7: [8],
        8: [7]
    }

    link_mat = create_link_matrix(links)

    print(f"Is link matrix column stochastic: {is_column_stochastic(link_mat)}\n")

    print(link_mat, "\n")

    eigVals, eigVecs = np.linalg.eig(link_mat)

    # indexes of the eigspaces with eigval = 1
    # decided to use .isclose due to possible rounding errors
    idx = np.where(np.isclose(eigVals, 1))

    print(f"dim(V1(A)) = {len(idx[0])}.")

    if len(idx[0]) >= 3:
        print("dim(V1(A)) equals (or exceeds) the number of the components in the web.")
    else:
        print("dim(V1(A)) does not equal the number of components in the web.")

