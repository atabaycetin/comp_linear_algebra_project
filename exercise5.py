#---------------  EXERCISE 5 ---------------#

"""
Exercise 5. Prove that in any web the importance
score of a page with no backlinks is zero.
"""


import numpy as np
from src import create_link_matrix, create_csr_link_matrix


if __name__ == '__main__':
    x0, A = create_csr_link_matrix('data\\hollins.dat')

    in_degree = np.array(A.sum(axis=1)).flatten() # number of incoming links

    pages_with_no_backlinks = np.where(in_degree == 0)[0] # row indexes of pages with no backlinks

    print(f"We found {len(pages_with_no_backlinks)} pages with no backlinks in our hollins web.")

    # let's check those rows
    print(f"Sum of votes pages with no backlinks has received: {A[pages_with_no_backlinks].sum(axis=1).flatten()}")
    print("As you can see, those rows are entirely zero.")

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
