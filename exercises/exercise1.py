#---------------  EXERCISE 1 ---------------#
"""
Text: Suppose the people who own page 3 in the web of
Figure 1 are infuriated by the fact that its importance score,
computed using formula (2.1), is lower than the score of page 1.
In an attempt to boost page 3’s score, they create a page 5 that
links to page 3; page 3 also links to page 5. Does this boost
page 3’s score above that of page 1?
"""

from src import create_link_matrix, cal_importance_score

if __name__ == "__main__":
    links = {
        1: [2, 3, 4],
        2: [3, 4],
        3: [1],
        4: [1, 3]
    }

    A = create_link_matrix(links)

    scores_A = cal_importance_score(A)

    # add page 5 and new links
    new_links = {k: v.copy() for k, v in links.items()}
    new_links[5] = [3]
    new_links[3].append(5)

    new_A = create_link_matrix(new_links)

    scores_new_A = cal_importance_score(new_A)

    if scores_new_A[2] > scores_new_A[0]:
        print("It worked. They did boost page 3's score")
    else:
        print("They can keep being infuriated")

