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

    # The file structure is:
    # Header
    # n_pages lines of 'ID URL'
    # n_links lines of 'SOURCE TARGET'

    # Skip header (1 line) and page definitions (n_pages lines)
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
    # Initial vector x0 (uniform)
    x = np.ones(n_pages) / n_pages
    x0_start = x.copy()

    # --- OPTIMIZATION 1: Efficient Matrix-Vector Product ---
    # We simplify the math to avoid creating dense arrays of 1s.
    # Original: x_new = (1 - m) * (Ax + dangle_vec) + m * teleport_vec
    # Optimized: x_new = (1 - m) * Ax + scalar_value

    def matvec(v):
        # 1. Sparse multiplication
        Ax = A @ v

        # 2. Calculate the scalar correction for dangling nodes + teleportation
        # Mass from dangling nodes = sum(v[dangling_indices])
        dangling_mass = np.sum(v[danglings])

        # Combined scalar to add to every element:
        # ((1-m) * (dangling_mass/N)) + (m * (sum(v)/N))
        # Note: inside the solver, sum(v) might not be exactly 1, so we keep it.
        scalar_add = ((1 - m) * dangling_mass / n_pages) + (m * np.sum(v) / n_pages)

        # Numpy broadcasts the scalar addition efficiently
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

    # Pre-calculate initial error for the ratio calculation
    prev_error = np.sum(np.abs(x - q))

    for k in range(1, max_k + 1):
        # --- OPTIMIZATION 2: Loop without allocations ---

        # 1. Sparse Matrix Multiply
        Ax = A @ x

        # 2. Compute scalar constant for this iteration
        # Since x is a probability vector, sum(x) is theoretically 1.0,
        # but we use the actual sum(x) or 1.0 depending on precision needs.
        dangle_sum = np.sum(x[danglings])

        # Combine the dangling part and the teleport part into one scalar
        # Term 1: (1-m) * (dangle_sum / N)
        # Term 2: m * (1.0 / N)
        scalar_correction = ((1 - m) * dangle_sum + m) / n_pages

        # 3. Update x
        # (1-m)Ax + scalar
        x = (1 - m) * Ax + scalar_correction

        # 4. Error Calculation
        current_error = np.sum(np.abs(x - q))

        if k in steps:
            ratio = current_error / prev_error if prev_error > 1e-16 else 0.0
            print(f"{k:<5} | {current_error:<25.8e} | {ratio:.4f}")

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
    print(f"c = max |1 - 2 * min(Mij)| = {c:.8f}")

    # Calculate second largest eigenvalue
    print("Computing second largest eigenvalue...")
    vals = scipy.sparse.linalg.eigs(M_op, k=2, which='LM', return_eigenvectors=False)
    # vals are unsorted complex numbers. Sort by magnitude.
    vals_sorted = sorted(np.abs(vals), reverse=True)
    lambda2 = vals_sorted[1]

    print(f"|lambda_2| = {lambda2:.8f}")


if __name__ == "__main__":
    # 1. Parse Data
    n_pages, links = parse_hollins('./data/hollins.dat')

    # 2. Create Matrices
    A = create_csr_link_matrix(n_pages, links)
    danglings = get_dangling_nodes(n_pages, links)

    # 3. Solve Exercise 14
    # Note: m=0.15 is the teleport probability (often denoted as alpha=0.85 in damping context)
    M_op = iterative_pagerank(A, n_pages, danglings, m=0.15, steps=[1, 5, 10, 50])

    # 4. Calculate Parameters
    calculate_c_and_lambda2(M_op, n_pages, m=0.15)