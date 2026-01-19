#---------------  EXERCISE 11 ---------------#
"""
Exercise 11. Consider again the web in Figure 2.1, with the addition of a page 5 that links to
page 3, where page 3 also links to page 5. Calculate the new ranking by finding the eigenvector of
M (corresponding to λ = 1) that has positive components summing to one. Use m = 0.15.
"""

import numpy as np
from exercises.src import create_link_matrix, cal_importance_score, figure21_links

new_links = {k: v.copy() for k, v in figure21_links.items()}
new_links.get(3).append(5)
new_links[5] = [3]

linkmatrix = create_link_matrix(new_links)

S = np.ones(linkmatrix.shape) * (1 / len(linkmatrix))
m = 0.15
M = (1 - m) * linkmatrix + m * S

if __name__ == "__main__":
    print(cal_importance_score(M))