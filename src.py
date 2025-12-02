import numpy as np

figure21_links = {
    1: [2, 3, 4],
    2: [3, 4],
    3: [1],
    4: [1, 3]
}

figure22_links = {
    1: [2],
    2: [1],
    3: [4],
    4: [3],
    5: [3, 4]
}

def create_link_matrix(links_list):
    n = len(links_list)

    link_matrix = np.zeros((n, n), dtype=np.float64)

    for j, link in links_list.items():
        # I want to handle dangling nodes in the simplest way possible.
        # Let's just distribute its vote to all pages in the web
        if len(link) == 0:
            weight = 1.0 / n
            for i in range(1, n + 1):
                link_matrix[i - 1, j - 1] = weight
        else:
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