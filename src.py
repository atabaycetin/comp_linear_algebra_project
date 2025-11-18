import numpy as np

def create_link_matrix(links_list):
    """
    speed up the creation of a link matrix
    because I can't be bothered to create a
    for loop every time links are updated
    """

    n = len(links_list)

    link_matrix = np.zeros((n, n), dtype=np.float64)

    for j, link in links_list.items():
        # no need to deal with dangling nodes for now because this is a simple exercise
        if len(link) == 0:
            raise ValueError(f"Page {j} has no outgoing links (dangling node).")
        weight = 1.0 / len(link)
        for i in link:
            link_matrix[i - 1, j - 1] += weight

    return link_matrix

def is_column_stochastic(link_matrix, tol=1e-12):
    non_negative = np.all(link_matrix >= -tol)
    columns_sum_to_one = np.allclose(link_matrix.sum(axis=0), 1, atol=tol)
    return non_negative and columns_sum_to_one

def cal_importance_score(link_matrix):
    n = link_matrix.shape[0]
    B = link_matrix - np.eye(n)
    B[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    x = np.linalg.solve(B, b)

    return x