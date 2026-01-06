import numpy as np
import scipy as sp

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
            continue
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

def create_csr_link_matrix(file_path):
    """
    This function is created to handle the provided web structure within the file "hollins.dat"
    :param file_path: path of the hollins.dat file
    :return: x0 (importance scores vector), A (CSR Link Matrix)
    """
    with open(file_path, "r") as f:
        header = f.readline().split()
        n_pages = int(header[0])
        # skip the url lines
        for _ in range(n_pages):
            next(f)

        sources = []
        targets = []

        seen_links = set()

        for line in f:
            parts = line.split()
            if len(parts) == 2:
                src = int(parts[0]) - 1
                dst = int(parts[1]) - 1

                if src == dst: continue

                if (src, dst) in seen_links: continue

                sources.append(src)
                targets.append(dst)
                seen_links.add((src, dst))

        data = np.ones(len(sources), dtype=float)  # will handle the out-degrees later

        A = sp.sparse.csr_matrix((data, (targets, sources)), shape=(n_pages, n_pages))  # source target reverse

        """
        The source and the target arrays are swapped above because csr normally creates an adjacency
        matrix, we want to create the link matrix instead, where the rows represent incoming links and 
        columns represent outgoing links.
        """

        out_degrees = np.array(A.sum(axis=0)).flatten()

        danglings = np.where(out_degrees == 0)[0]  # apparently np.where returns a tuple thus the [0] indexing

        # weight adjustment (nj)
        A.data /= out_degrees[A.indices]  # now we have a correct link matrix

        # now the iterative solution
        m = 0.15
        x0 = np.ones(n_pages) * (1 / n_pages)
        s = np.ones(n_pages) * (1 / n_pages)
        for i in range(50):
            dangling_mass = (1 - m) * (x0[danglings].sum())  # distributing the dangling probability uniformly
            x0 = (1 - m) * (A @ x0) + (m * s)
            x0 += dangling_mass / n_pages

    return x0, A