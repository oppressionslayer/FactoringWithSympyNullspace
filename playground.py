import numpy as np
import numba as nb

@nb.njit
def gauss_elim(M):
    nrows, ncols = M.shape
    pivot_rows, pivot_cols = [], []
    for i in range(nrows):
        # Find pivot in row i
        row = M[i]
        cols = np.where(row == 1)[0]  # Efficiently find indices of 1s
        if cols.size > 0:
            pivot_col = cols[0]
            pivot_cols.append(pivot_col)
            pivot_rows.append(i)
            for j in range(nrows):
                if j != i and M[j, pivot_col] == 1:
                    M[j] = (M[j] + M[i]) % 2  # Vectorized row operation
        else:
            continue

    # Identify free columns (those not chosen as pivot columns)
    free_cols = [col for col in range(ncols) if col not in pivot_cols]

    # Find potential solutions based on free columns
    sol_rows = [row for row in free_cols]

    if not sol_rows:
        print("No solution found. Need more smooth numbers.")
    else:
        print(f"Found {len(sol_rows)} potential solutions")

    # This is a simplified conceptual approach; further refinement might be needed
    # to directly extract the null space vectors from the reduced matrix.