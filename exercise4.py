#---------------  EXERCISE 4 ---------------#

"""
Exercise 4. In the web of Figure 2.1, remove the link from page 3 to page 1. In the resulting
web page 3 is now a dangling node. Set up the corresponding substochastic matrix and find its largest
positive (Perron) eigenvalue. Find a non-negative Perron eigenvector for this eigenvalue, and scale
the vector so that components sum to one. Does the resulting ranking seem reasonable?
"""

import numpy as np
from src import create_link_matrix, cal_importance_score, figure21_links

if __name__ == "__main__":

    new_links = {k: v.copy() for k, v in figure21_links.items()}; new_links[3].remove(1)

    link_mat = create_link_matrix(new_links)

    eigvals, eigvecs = np.linalg.eig(link_mat)

    print(f"Current web:\n{new_links}\n")
    print(f"Current link matrix:{link_mat}\n")

    idx_largest_pos = np.argsort(eigvals.real)[::-1][0]

    eigval_perron = eigvals[idx_largest_pos]
    eigvec_perron = eigvecs[:, idx_largest_pos]

    print(f"Perron eigenvalue: {eigval_perron}")
    print(f"Perron eigenvector: {eigvec_perron}\n")

    scaled_perronvec = eigvec_perron.real / eigvec_perron.real.sum()
    print(f"Scaled (Normalized) Perron eigenvector: {scaled_perronvec}\n")
    print(f"Page 3 has the highest importance score, hence the result is not reasonable.")