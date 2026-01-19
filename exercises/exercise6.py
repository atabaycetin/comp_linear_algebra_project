#---------------  EXERCISE 6 ---------------#

"""
Exercise 6. Implicit in our analysis up to this point is the assertion that the manner in which
the pages of a web W are indexed has no effect on the importance score assigned to any given page.
Prove this, as follows: Let W contains n pages, each page assigned an index 1 through n, and let
A be the resulting link matrix. Suppose we then transpose the indices of pages i and j (so page i is
now page j and vice versa). Let A˜ be the link matrix for the relabelled web.
  • Argue that A_tilde = PAP, where P is the elementary matrix obtained by transposing rows i and
    j of the n × n identity matrix. Note that the operation A → PA has the eﬀect of swapping
    rows i and j of A, while A → AP swaps columns i and j. Also, P2 = I, the identity
    matrix.
  • Suppose that x is an eigenvector for A, so Ax = λx for some λ. Show that y = Px is an
    eigenvector for A˜ with eigenvalue λ.
  • Explain why this shows that transposing the indices of any two pages leaves the importance
    scores unchanged, and use this result to argue that any permutation of the page indices leaves
    the importance scores unchanged.
"""


import numpy as np
from exercises.src import create_link_matrix, figure21_links


if __name__ == '__main__':
    # construct the link matrix A
    A = create_link_matrix(figure21_links)

    # pick arbitrary i and j
    i, j = np.random.choice(A.shape[0], size=2, replace=False)

    # construct the matrix P
    P = np.eye(A.shape[0]); P[[i, j], :] = P[[j, i], :]

    # modified A by multiplying it with Ps
    A_mod = P @ A @ P

    # construct the answer to compare with our result
    A_tilde = A.copy(); A_tilde[[i, j], :] = A_tilde[[j, i], :]; A_tilde[:, [i, j]] = A_tilde[:, [j, i]]

    print("Modified A as A_mod = PAP:")
    print(A_mod)

    print("\nActual A_tilde:")
    print(A_tilde)

    print(f"\nIs A_mod equal to A_tilde: {np.allclose(A_mod, A_tilde)}")

    if np.allclose(A_mod, A_tilde):
        print("We have showed that both matrices are equal\n")

    # calculate eigenvalues and eigenvectors of A
    eigval_A, eigvec_A = np.linalg.eig(A)

    # pick an arbitrary one to check
    eig_lambda = eigval_A[0]
    x_ = eigvec_A[:, 0]

    # supposed eigenvector (to be shown)
    y = P @ x_

    # let's see if the effects of the matrix A_mod matches (see if it is its eigenvalue)
    check_1 = A_mod @ y
    check_2 = eig_lambda * y

    print(f"Eigenvalue lambda: {eig_lambda}")
    print(f"A_tilde * y: {check_1}")
    print(f"lambda * y: {check_2}\n")

    # Check if they are equal
    print(f"Is y = Px an eigenvector of A_tilde: {np.allclose(check_1, check_2)}")

    print("\nFinal Comments:\nAs probably we can intuitively conclude, the importance scores of the pages in a web\n"
          "do not depend on how the pages are indexed, but rather depend only on the link structure.\n"
          "In the first part, we perfomed permutation on the link matrix, and we know that permutations\n"
          "do not change the eigenvalues, it rather permutes the eigenvector entries. (bc perms are products of row swaps)\n"
          "At last, let's formalize this phenomenon mathematically:\n"
          "We have Ã = PAP, Ax = λx, and we will define y = Px:\n"
          "Ãy = (PAP)(Px) = PA(P²)x = PAx = P(λx) = λ(Px) = λy\n"
          "End of Comment")