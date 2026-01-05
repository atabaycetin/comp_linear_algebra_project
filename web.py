import numpy as np
import scipy as sp

#this file is to handle the provided web structure with the file "hollins.dat"
def csr_linkmatrix():
    with open("hollins.dat", "r") as f:
        header = f.readline().split()
        n_pages = int(header[0])
        #skip the url lines
        for _ in range(n_pages):
            next(f)
        
        sources = []
        targets = []

        for line in f:
            parts = line.split()
            if len(parts)==2:
                src = int(parts[0])-1
                dst = int(parts[1])-1
                sources.append(src)
                targets.append(dst)
        
        data = np.ones(len(sources), dtype=float) #will handle the out-degrees later

        A = sp.sparse.csr_matrix((data,(targets, sources)),shape = (n_pages, n_pages)) #source target reverse

        """
        The source and the target arrays are swapped above because csr normally creates an adjacency
        matrix, we want to create the link matrix instead, where the rows represent incoming links and 
        columns represent outgoing links.
        """

        out_degrees = np.array(A.sum(axis=0)).flatten()

        danglings = np.where(out_degrees == 0)[0] #apparently np.where returns a tuple thus the [0] indexing
        
        #weight adjustment (nj)
        A.data /= out_degrees[A.indices] #now we have a correct link matrix

        #now the iterative solution
        m=0.15
        x0 = np.ones(n_pages)*(1/n_pages)
        s = np.ones(n_pages)*(1/n_pages)
        for i in range(50):
            dangling_mass = (1-m)*(x0[danglings].sum())#distributing the dangling probability uniformly
            x0 = (1-m)*(A@x0) + (m*s)
            x0 += dangling_mass/n_pages
            
    return x0

print(f"{csr_linkmatrix()}, {csr_linkmatrix().sum()}")

