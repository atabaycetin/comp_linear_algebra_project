#---------------  EXERCISE 5 ---------------#

"""
Exercise 5. Prove that in any web the importance
score of a page with no backlinks is zero.
"""


import numpy as np
from src import create_link_matrix, cal_importance_score, figure21_links


if __name__ == '__main__':
    links = {
        1: [2, 3],
        2: [3],
        3: [2]
    }

    print(f"Web Structure: {links}")

    link_mat = create_link_matrix(links)

    print(f"Link Matrix A:\n{link_mat}")
    print("You can see that first row is entirely zero.")
    print("This confirms that no page votes for page 1 (page 1 has no backlinks).")

    eigvals, eigvecs = np.linalg.eig(link_mat)

    idx_largest_pos = np.argsort(eigvals.real)[::-1][0]

    eigval_perron = eigvals[idx_largest_pos]
    eigvec_perron = eigvecs[:, idx_largest_pos]

    scaled_perronvec = eigvec_perron.real / eigvec_perron.real.sum()
    print(f"Normalized Perron eigenvector: {scaled_perronvec}\n")
    print("You can see that first index, the importance score of page 1, is zero.\n"
          "Hence, we proved that the importance score of a page with no backlinks is indeed zero.")