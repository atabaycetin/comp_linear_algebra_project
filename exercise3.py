#---------------  EXERCISE 2 ---------------#

"""
Exercise 3. Add a link from page 5 to page 1
in the web of Figure 2. The resulting web,
considered as an undirected graph, is connected.
What is the dimension of V1(A)?
"""

import numpy as np
from src import create_link_matrix, is_column_stochastic, cal_importance_score

if __name__ == "__main__":
    links = {
        1: [2],
        2: [1],
        3: [4],
        4: [3],
        5: [3, 4]
    }

    org_link_mat = create_link_matrix(links) # original figure 2 link matrix

    links[5].append(1); new_link_mat = create_link_matrix(links)

    eigval, eigvec = np.linalg.eig(new_link_mat)

    idx = np.where(np.isclose(eigval, 1))

    print(f"After adding a new link from page 5 to page 1,\n"
          f"Dimension of V1(A) is: {len(idx[0])}")
