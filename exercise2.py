#---------------  EXERCISE 2 ---------------#

"""
Exercise 2. Construct a web consisting of three or more subwebs
and verify that dim(V1(A)) equals (or exceeds) the number of
the components in the web.
"""

from src import create_link_matrix, is_column_stochastic, cal_importance_score

if __name__ == "__main__":
    links = {
        1: [2, 3],
        2: [1, 3],
        3: [2],
        4: [5, 6],
        5: [4, 6],
        6: [5, 4],
        7: [8],
        8: [7]
    }

    link_mat = create_link_matrix(links)

    print(f"Is link matrix column stochastic: {is_column_stochastic(link_mat)}\n")

    print(link_mat)
