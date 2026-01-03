#---------------  EXERCISE 9 ---------------#
"""
Show that a page with no backlinks is given importance score m/n by formula (3.2).
"""

import numpy as np
from src import create_link_matrix, cal_importance_score, figure21_links as figure

#We will modify a figure we already have and add a node without any backlinks so we can work on it

modified = {k: v.copy() for k, v in figure.items()}
modified[5] = [3]

"""
Now, the figure has a node 5 pointing towards node 3 (so that it is not a dangling node,
it could point towards any node, node 3 is arbitrary), without any backlinks.
"""

linkmatrix = create_link_matrix(modified)
S = np.ones(linkmatrix.shape) * (1 / len(linkmatrix))
m = 0.15
M = (1 - m) * linkmatrix + m * S
result_M = cal_importance_score(M)

print(f"The importance score of the node without any backlinks: {result_M[4]:.2f}")
print("It matches 'm/n' which is 0.15/5 = 0.03 in this case")