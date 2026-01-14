#---------------  EXERCISE 3 ---------------#

"""
Exercise 3. Add a link from page 5 to page 1
in the web of Figure 2. The resulting web,
considered as an undirected graph, is connected.
What is the dimension of V1(A)?
"""

import numpy as np
from src import create_link_matrix, figure22_links

if __name__ == "__main__":

    org_link_mat = create_link_matrix(figure22_links) # original figure 2 link matrix

    new_links = {k: v.copy() for k, v in figure22_links.items()}

    new_links[5].append(1); new_link_mat = create_link_matrix(new_links)

    eigval, eigvec = np.linalg.eig(new_link_mat)

    idx = np.where(np.isclose(eigval, 1))

    print(f"After adding a new link from page 5 to page 1,\n"
          f"Dimension of V1(A) is: {len(idx[0])}")
