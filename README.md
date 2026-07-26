# Large-Scale Computational Linear Algebra

A collection of Python solutions to selected computational linear algebra
exercises centered on web ranking and PageRank. The examples progress from
small, dense link matrices to a sparse computation on a 6,012-page web graph.

The repository explores:

- column-stochastic and substochastic link matrices;
- Perron eigenvalues and eigenvectors;
- disconnected subwebs and the eigenspace associated with $\lambda = 1$;
- invariance of rankings under page reindexing;
- the Google matrix and teleportation;
- power iteration and convergence rates; and
- sparse matrix storage and computation with SciPy.

## Project structure

```text
.
├── data/
│   └── hollins.dat           # 6,012-page web graph
├── exercises/
│   ├── src.py                # Shared link-matrix and ranking helpers
│   ├── exercise1.py
│   ├── ...
│   ├── exercise14_hollins.py # Sparse convergence experiment
│   └── exercise16.py
├── requirements.txt
└── LICENSE
```

The exercise numbering follows the source problem set, so the files are not
numbered consecutively.

## Exercises

| File | Main topic |
| --- | --- |
| `exercise1.py` | How adding a mutually linked page changes importance scores |
| `exercise2.py` | Disconnected subwebs and the dimension of $V_1(A)$ |
| `exercise3.py` | The effect of connecting previously separate subwebs |
| `exercise4.py` | Dangling nodes, substochastic matrices, and Perron eigenpairs |
| `exercise5.py` | Pages with no backlinks, including a check on the Hollins graph |
| `exercise6.py` | Ranking invariance under a permutation of page indices |
| `exercise9.py` | The $m/n$ score for a page with no backlinks |
| `exercise11.py` | Ranking with the Google matrix and $m = 0.15$ |
| `exercise12.py` | Comparing rankings produced by the link and Google matrices |
| `exercise13.py` | Google-matrix ranking for a web with multiple subwebs |
| `exercise14.py` | Power-iteration error, convergence bounds, and the second eigenvalue |
| `exercise14_hollins.py` | Sparse version of the convergence experiment |
| `exercise16.py` | Defectiveness and diagonalizability of a PageRank matrix |

## Getting started

The pinned NumPy and SciPy versions require Python 3.11 or later. Create a
virtual environment and install the dependencies:

```bash
git clone https://github.com/atabaycetin/large-scale-computational-linear-algebra.git
cd large-scale-computational-linear-algebra

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the examples

Run scripts from the repository root so that the relative path to
`data/hollins.dat` resolves correctly:

```bash
python exercises/exercise1.py
python exercises/exercise14.py
python exercises/exercise14_hollins.py
```

Each script can be run directly and prints its matrices, rankings, or
convergence measurements to the terminal.

## Shared utilities

`exercises/src.py` provides the common building blocks used by the examples:

- `create_link_matrix(links_list)` builds a dense link matrix from a dictionary
  of outgoing links;
- `is_column_stochastic(link_matrix)` checks non-negativity and unit column
  sums;
- `cal_importance_score(link_matrix)` solves for a normalized stationary
  importance vector; and
- `create_csr_link_matrix(file_path)` loads the Hollins graph into SciPy's CSR
  sparse format and performs damped PageRank iteration.

Page identifiers in the small graph dictionaries are one-based, matching the
exercise statements.

## Hollins dataset

The bundled `data/hollins.dat` file describes a web graph with 6,012 pages and
23,875 recorded links. The sparse examples remove self-links and duplicate
edges, normalize outgoing votes, and distribute dangling-node mass uniformly.
Keeping this graph in CSR format avoids constructing a dense 6,012 × 6,012
matrix.

## License

This project is available under the [MIT License](LICENSE).
