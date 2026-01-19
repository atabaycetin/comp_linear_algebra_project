#---------------  EXERCISE 12 ---------------#
"""
Exercise 12. Add a sixth page that links to every page of the web in the previous exercise, but
to which no other page links. Rank the pages using A, then using M with m = 0.15, and compare
the results.
"""
#this exercise focuses on a node without any backlinks
#since this exercise utilizes the web from the exercise-11, we will import the links from there


import numpy as np
from exercises.src import create_link_matrix, cal_importance_score
from exercise11 import new_links as old_links


if __name__ == "__main__":
    new_links = {k: v.copy() for k, v in old_links.items()}
    new_links[6] = [1, 2, 3, 4, 5]

    # the linkmatrix is "A" here
    linkmatrix = create_link_matrix(new_links)

    S = np.ones(linkmatrix.shape) * (1 / len(linkmatrix))
    m = 0.15
    M = (1 - m) * linkmatrix + m * S

    result_A = cal_importance_score(linkmatrix)
    result_M = cal_importance_score(M)
    print(f"The ranking according to matrix A: {result_A}\nThe ranking according to matrix M: {result_M}")