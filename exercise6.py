import numpy as np
from pywin.mfc.afxres import AFX_IDP_SQL_ILLEGAL_MODE

from src import create_link_matrix, cal_importance_score, figure21_links


if __name__ == '__main__':

    np.random.seed(42)

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

    print("We have proven that both matrices are equal\n")

    # calculate eigenvalues and eigenvectors of A
    eigval_A, eigvec_A = np.linalg.eig(A)

    # pick an arbitrary one to check
    eig_lambda = eigval_A[0].real
    x_ = eigvec_A[:, 0].real

    # supposed eigenvector (to be proven)
    y = P @ x_

    # let's see if the effects of the matrix A_mod matches (it is its eigenvalue)
    check_1 = A_mod @ y
    check_2 = eig_lambda * y

    print(f"Eigenvalue lambda: {eig_lambda}")
    print(f"A_tilde * y: {check_1}")
    print(f"lambda * y: {check_2}\n")

    # Check if they are equal
    print(f"Is y = Px an eigenvector of A_tilde: {np.allclose(check_1, check_2)}")