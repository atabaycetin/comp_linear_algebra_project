#---------------  EXERCISE 13 ---------------#
"""
Construct a web consisting of two or more subwebs and determine the ranking
given by formula (3.1).
"""
"""since we already have a web consisting of two subwebs, 
we will be using that (fig2.2) with a little modification"""

import numpy as np
from src import figure22_links, create_link_matrix, cal_importance_score

new_links = {k: v.copy() for k, v in figure22_links.items()}
# adding an extra node to "create" a different web
new_links[6] = [1, 2]
new_links.get(2).append(6)

global M

def main():
    global M, new_links

    linkmatrix = create_link_matrix(new_links)

    S = np.ones(linkmatrix.shape) * (1 / len(linkmatrix))
    m = 0.15
    M = (1 - m) * linkmatrix + m * S

if __name__ == "__main__":
    main()
    print(cal_importance_score(M))