import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def create_csr_and_setup(file_path):
    
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
            if len(parts) >= 2:
                src = int(parts[0]) - 1
                dst = int(parts[1]) - 1

                if src == dst: continue
                if (src, dst) in seen_links: continue

                sources.append(src)
                targets.append(dst)
                seen_links.add((src, dst))

        data = np.ones(len(sources), dtype=float)

        # Create A (Rows=Targets, Cols=Sources)
        A = sp.csr_matrix((data, (targets, sources)), shape=(n_pages, n_pages))

        out_degrees = np.array(A.sum(axis=0)).flatten()
        
        danglings = np.where(out_degrees == 0)[0]
        
        A.data /= out_degrees[A.indices]

    return A, danglings, n_pages

def solve_exercise_14(file_path):

    A, danglings, n_pages = create_csr_and_setup(file_path)
    
    m = 0.15
    s = np.ones(n_pages) * (1 / n_pages) 

    # We will compute True Eigenvector q with a library for Error Comparison

    def matvec(v):

        dangling_mass = (1 - m) * (v[danglings].sum())
        
        # Note: For the LinearOperator, we assume v sums to something arbitrary,
        # so we normalize the teleport/dangling addition by sum(v) to keep it linear.
        v_sum = np.sum(v)
        
        res = (1 - m) * (A @ v) 
        res += (dangling_mass / n_pages) 
        res += (m * v_sum / n_pages)     
        return res

    M_op = scipy.sparse.linalg.LinearOperator((n_pages, n_pages), matvec=matvec)
    
    print("Computing exact eigenvector q...")
    evals, evecs = scipy.sparse.linalg.eigs(M_op, k=1, which='LM', tol=1e-14)
    q = np.real(evecs[:, 0])
    q = q / np.sum(q) # Normalize q

    # Iterative Solution (Our Loop Implementation)
    
    x0 = np.ones(n_pages) * (1 / n_pages)
    
    steps_to_record = [1, 5, 10, 50]
    max_step = 50
    
    print(f"\n{'k':<5} | {'Error ||M^k x0 - q||_1':<25} | {'Ratio'}")
    print("-" * 55)
    
    prev_error = np.sum(np.abs(x0 - q))
    
    for k in range(1, max_step + 1):
        dangling_mass = (1 - m) * (x0[danglings].sum())
        x0 = (1 - m) * (A @ x0) + (m * s)
        x0 += dangling_mass / n_pages
        
        current_error = np.sum(np.abs(x0 - q))
        
        if k in steps_to_record:
            ratio = current_error / prev_error if prev_error > 1e-16 else 0.0
            print(f"{k:<5} | {current_error:<25.4e} | {ratio:.4e}")
            
        prev_error = current_error

    
    print("\n--- Constants ---")
    
    # c = max | 1 - 2 * min(Mij) |
    # The minimum value in the Google Matrix M is strictly determined by the teleportation factor
    # because sparse entries are 0. So min(Mij) = m/N.
    min_Mij = m / n_pages
    c = abs(1 - 2 * min_Mij)
    print(f"c = {c:.4e}")

    # Second largest eigenvalue
    vals = scipy.sparse.linalg.eigs(M_op, k=2, which='LM', return_eigenvectors=False)
    vals_sorted = sorted(np.abs(vals), reverse=True)
    lambda2 = vals_sorted[1]
    print(f"|lambda_2| = {lambda2:.4e}")

if __name__ == "__main__":
    solve_exercise_14('./data/hollins.dat')