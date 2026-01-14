#---------------  EXERCISE 9 ---------------#
"""
Show that a page with no backlinks is given importance score m/n by formula (3.2).
"""

import numpy as np
from src import create_link_matrix, create_csr_link_matrix, cal_importance_score, figure21_links as figure

if __name__ == '__main__':
    # We will modify a figure we already have and add a node without any backlinks so we can work on it

    modified = {k: v.copy() for k, v in figure.items()}
    modified[5] = [3]

    """
    Now, the figure has a node 5 pointing towards node 3 (so that it is not a dangling node,
    it could point towards any node, node 3 is arbitrary), without any backlinks.
    """

    linkmatrix = create_link_matrix(modified)
    S = np.ones(linkmatrix.shape) * (1 / len(linkmatrix))
    m = 0.15
    M = (1 - m) * linkmatrix + m * S
    result_M = cal_importance_score(M)

    # here we are doing the same visualization for the hollins web
    x0, _, no_backlinks, dangling_mass, n_pages = create_csr_link_matrix("../data/hollins.dat")
    example_node = x0[no_backlinks[0]]
    """
    But in this web, we handled dangling nodes via distributing their probability with random jumps.
    So the importance score of the nodes without backlinks is not only m/n but also 
    sum_of_dangling_nodes/n. So we will act as if there are no dangling nodes and substract their
    random jump distribution for this specific exercise.
    """
    example_node = example_node - (dangling_mass / n_pages)

    print(f"The importance score of the node without any backlinks: {result_M[4]:.2f}")
    print("It matches 'm/n' which is 0.15/5 = 0.03 in this case")
    print(f"The importance score of the node without any backlinks from hollins web: {example_node} ")
    print("It matches 'm/n' again, which is 0.15/6012 = 2.495e-05 in this case")