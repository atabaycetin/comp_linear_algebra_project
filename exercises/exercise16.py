#---------------  EXERCISE 16 ---------------#

"""
Consider the link matrix A (see from the figure below)
Show that M = (1− m)A + mS (all Sij = 1/3) is not diagonalizable for 0 ≤ m < 1.
"""


if __name__ == "__main__":

    import numpy as np
    from numpy.linalg import matrix_rank

    matrix_A = np.array([
            [0,   0.5, 0.5],
            [0,   0,   0.5],
            [1,   0.5, 0]])

    """
    The aim is to show the matrix M is not diagonalizable (defective).
    The defectiveness of the Matrix M depends entirely on the original matrix A because 
    the teleportation mechanism only fixes the ambiguity of having multiple winners (at λ=1)
    """
    eigvals = np.round(np.linalg.eigvals(matrix_A),4)
    unique_vals, counts = np.unique(eigvals, return_counts=True)
    for val, alg_mult in zip(unique_vals, counts):
        n = 3
        I = np.eye(n)
        temp_matrix = matrix_A - (val * I)
        r = matrix_rank(temp_matrix, tol=1e-5)
        geo_mult = n - r
        if geo_mult < alg_mult:
            print("Matrix A is defective.")
            print("Therefore, M = (1-m)A + mS maintains this defective structure")