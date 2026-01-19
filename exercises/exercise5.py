#---------------  EXERCISE 5 ---------------#

"""
Exercise 5. Prove that in any web the importance
score of a page with no backlinks is zero.
"""


import numpy as np
from exercises.src import create_link_matrix, create_csr_link_matrix


if __name__ == '__main__':
    # an example web
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

    print("\n" + "-" * 60 + "\n")
    print("Now, let's do the exercise on the Hollins dataset")
    _, A, _, _, _ = create_csr_link_matrix('../data/hollins.dat')

    in_degree = np.array(A.sum(axis=1)).flatten() # number of incoming links

    pages_with_no_backlinks = np.where(in_degree == 0)[0] # row indexes of pages with no backlinks

    print(f"We found {len(pages_with_no_backlinks)} pages with no backlinks in our Hollins web.\n")
    print(pages_with_no_backlinks)
    print("As you can see, those rows are entirely zero.\n")

    # let's check those rows
    print(f"Sum of votes the pages with no backlinks have received: {A[pages_with_no_backlinks].sum(axis=1).flatten()}")

    # now we choose a page to check its importance score
    test_page = pages_with_no_backlinks[0]

    # we will preallocate the importance scores vector (eigenvector)
    n_pages = A.shape[0]
    x = np.ones(n_pages) * (1 / n_pages)

    # here we are basically applying the first step of the power method
    # since the page with no backlinks will receive a total vote of 0,
    # there is no need to iterate until the scores stabilize. The first
    # iteration gives us what we need anyway.
    x_new = A @ x

    score = x_new[test_page]

    print(f"Page: {test_page} has an importance score of {score}")
    print("Hence, we proved that the importance score of a page with no backlinks is indeed zero.")
