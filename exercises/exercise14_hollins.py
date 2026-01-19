#---------------  EXERCISE 14 ---------------#
# We created a new file since the solution would be too long for a single file

"""
Exercise 14. For the web in Exercise 11, compute the values of ∥Mk x0−q∥1 and ∥Mk x0−q∥1
∥Mk−1 x0−q∥1
for k = 1, 5, 10, 50, using an initial guess x0 not too close to the actual eigenvector q (so that you
can watch the convergence). Determine c = max1≤j≤n |1− 2 min1≤i≤n Mij | and the absolute value
of the second largest eigenvalue of M.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


def parse_hollins(filename):
    """Parses the hollins.dat file to extract links."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Read header
    header = lines[0].strip().split()
    n_pages = int(header[0])
    n_links_expected = int(header[1])

    print(f"Dataset info: {n_pages} pages, {n_links_expected} links declared.")

    links = {}

    # we skip the header
    start_line = 1 + n_pages

    edge_lines = lines[start_line:]

    for line in edge_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            src = int(parts[0])
            dst = int(parts[1])

            # Store in links dict (1-based index from file)
            if src not in links:
                links[src] = []
            links[src].append(dst)
        except ValueError:
            continue

    return n_pages, links


def create_csr_link_matrix(n_pages, links_dict):
    """
    Creates a column-stochastic sparse link matrix from the links dictionary.
    Rows represent incoming links (targets), columns represent outgoing links (sources).
    """
    sources = []
    targets = []
    data = []

    # Build coordinate lists
    for src, dst_list in links_dict.items():
        if len(dst_list) == 0:
            continue

        # Determine weight for this source (1 / out_degree)
        weight = 1.0 / len(dst_list)

        for dst in dst_list:
            # Adjust to 0-based indexing for matrix
            sources.append(src - 1)
            targets.append(dst - 1)
            data.append(weight)

    # Create Compressed Sparse Row matrix
    # Shape is (n_pages, n_pages)
    # sources are columns, targets are rows
    A = sp.csr_matrix((data, (targets, sources)), shape=(n_pages, n_pages))

    return A


def get_dangling_nodes(n_pages, links_dict):
    """Identifies nodes with no outgoing links."""
    all_nodes = set(range(1, n_pages + 1))
    nodes_with_outlinks = set(links_dict.keys())
    dangling = list(all_nodes - nodes_with_outlinks)
    # Convert to 0-based index
    return np.array([d - 1 for d in dangling])


def iterative_pagerank(A, n_pages, danglings, m=0.15, steps=[1, 5, 10, 50]):
    """
    Computes PageRank iterates and errors.
    A: Sparse link matrix (column stochastic for non-dangling nodes)
    m: Teleportation probability (1 - damping factor)
    """
    # Initial random vector x0 
    x = np.ones(n_pages) / n_pages
    x0_start = x.copy()

    # We will be using the optimized equatio to avoid dense matrices: x_new = (1 - m) * Ax + scalar_value

    def matvec(v):

        Ax = A @ v

        # Calculate the scalar correction for dangling nodes
        dangling_mass = np.sum(v[danglings])

        # Combined scalar to add to every element:
        # ((1-m) * (dangling_mass/N)) + (m * (sum(v)/N))
        scalar_add = ((1 - m) * dangling_mass / n_pages) + (m * np.sum(v) / n_pages)
        return (1 - m) * Ax + scalar_add

    print("Computing steady state q...")
    M_op = scipy.sparse.linalg.LinearOperator((n_pages, n_pages), matvec=matvec)

    # Find eigenvector for eigenvalue 1
    evals, evecs = scipy.sparse.linalg.eigs(M_op, k=1, which='LM', tol=1e-12)
    q = np.real(evecs[:, 0])
    q = q / np.sum(q)

    # Now run the specific iterations requested
    x = x0_start.copy()
    max_k = max(steps)

    print(f"{'k':<5} | {'Error ||M^k x0 - q||_1':<25} | {'Ratio'}")
    print("-" * 45)

    # First we calculate initial error for the ratio calculation
    prev_error = np.sum(np.abs(x - q))

    for k in range(1, max_k + 1):

        Ax = A @ x

        # Compute scalar constant for this iteration
        dangle_sum = np.sum(x[danglings])

        # Combine the dangling part and the teleport part into one scalar
        scalar_correction = ((1 - m) * dangle_sum + m) / n_pages

        # Update x
        # (1-m)Ax + scalar
        x = (1 - m) * Ax + scalar_correction

        # Error Calculation
        current_error = np.sum(np.abs(x - q))

        if k in steps:
            ratio = current_error / prev_error if prev_error > 1e-16 else 0.0
            print(f"{k:<5} | {current_error:<25.4e} | {ratio:.4e}")

        prev_error = current_error

    return M_op


def calculate_c_and_lambda2(M_op, n_pages, m):
    """Calculates c and the second largest eigenvalue."""
    # c = max_j | 1 - 2 * min_i M_ij |
    # For a web matrix with teleportation m, M_ij >= m/n
    # The minimum value in any column is exactly m/n (unless the graph is fully connected which is rare)
    min_Mij = m / n_pages
    c = abs(1 - 2 * min_Mij)

    print("-" * 45)
    print(f"c = max |1 - 2 * min(Mij)| = {c:.4e}")

    # Calculate second largest eigenvalue
    print("Computing second largest eigenvalue...")
    vals = scipy.sparse.linalg.eigs(M_op, k=2, which='LM', return_eigenvectors=False)
    # vals are unsorted complex numbers. Sort by magnitude.
    vals_sorted = sorted(np.abs(vals), reverse=True)
    lambda2 = vals_sorted[1]

    print(f"|lambda_2| = {lambda2:.4e}")


if __name__ == "__main__":
    n_pages, links = parse_hollins('./data/hollins.dat')

    A = create_csr_link_matrix(n_pages, links)
    danglings = get_dangling_nodes(n_pages, links)

    # Note: m=0.15 is the teleport probability (often denoted as alpha=0.85 in damping context)
    M_op = iterative_pagerank(A, n_pages, danglings, m=0.15, steps=[1, 5, 10, 50])

    calculate_c_and_lambda2(M_op, n_pages, m=0.15)