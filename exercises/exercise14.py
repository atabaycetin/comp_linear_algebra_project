#---------------  EXERCISE 14 ---------------#
"""
Exercise 14. For the web in Exercise 11, compute the values of ∥Mk x0−q∥1 and ∥Mk x0−q∥1
∥Mk−1 x0−q∥1
for k = 1, 5, 10, 50, using an initial guess x0 not too close to the actual eigenvector q (so that you
can watch the convergence). Determine c = max1≤j≤n |1− 2 min1≤i≤n Mij | and the absolute value
of the second largest eigenvalue of M.
"""


import numpy as np
from exercise11 import M, linkmatrix
from src import cal_importance_score

def iterativeapproach(link_matrix):
    m = 0.15
    s = np.ones(len(link_matrix))*(1/len(link_matrix))
    x0 = np.ones(len(link_matrix))*(1/len(link_matrix))
    xvalues = []
    for k in range(50):
        if(k==0 or k==4 or k==9 or k==49):
            xvalues.append(x0)#this gives us M^(k-1)x0
        x0 = (1-m)*link_matrix@x0 + m*s
    return xvalues

if __name__ == "__main__":
    q = cal_importance_score(M)
    print(f"{'k':<5} | {'Error ||M^k x0 - q||_1':<25} | {'Ratio (Error_k / Error_k-1)':<25}")
    print("-" * 60)
    nparray = np.array(iterativeapproach(linkmatrix))
    k = [1, 5, 10, 50]
    count = 0
    for x0 in nparray:
        error_subk = np.sum(np.abs(x0 - q))
        error_k = np.sum(np.abs(M @ x0 - q))
        ratio = error_k / error_subk if error_subk > 1e-15 else 0.0
        print(f"{k[count]:<5} | {error_k:<25.4e} | {ratio:<25.4e}")
        count += 1

    print("\n--- Theoretical Constants ---")
    # Calculate c = max_j | 1 - 2 * min_i(M_ij) |
    # Note: min_i(M_ij) is the minimum value in column j
    min_col_values = np.min(M, axis=0)
    c_values = np.abs(1 - 2 * min_col_values)
    c = np.max(c_values)
    print(f"Bound constant c (Proposition 4): {c:.4e}")
    # Calculate second largest eigenvalue magnitude
    eigenvalues, eigenvectors = np.linalg.eig(M)
    sorted_eig_mags = np.sort(np.abs(eigenvalues))[::-1]  # Sort descending
    lambda_2 = sorted_eig_mags[1]  # The 2nd one (index 1)
    print(f"|lambda_2| (Actual convergence rate): {lambda_2:.4e}")