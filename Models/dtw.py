import numpy as np
from numba import njit

# from classifier.nearest_neighbours.distances import EPSILON, square_cost, cost_function

EPSILON = 10e-12


def square_cost(a, b):
    d = a - b
    return d ** 2


def cost_function(a, b, ge):
    if ge == 2:
        return square_cost(a, b)
    elif ge == 0.5:
        return np.sqrt(a, b)
    elif ge != 1:
        d = np.abs(a - b)
        return d ** ge
    else:
        return np.abs(a - b)

'''
@njit("float64(float64[:],float64[:],float64)", cache=True, fastmath=True)
def dtw(lines, cols, cutoff):
    # Ensure that lines are longer than columns
    if lines.shape[0] < cols.shape[0]:
        swap = lines
        lines = cols
        cols = swap

    # --- --- --- Declarations
    nblines = lines.shape[0]
    nbcols = cols.shape[0]

    # Setup buffers - no extra initialization required - border condition managed in the code.
    buffers = np.zeros((2 * nbcols,), dtype=np.float64)
    c = 0  # Start of current line in buffer - account for the extra cell
    p = nbcols  # Start of previous line in buffer - account for two extra cells

    # Line & columns indices
    i = 0
    j = 0

    # Cost accumulator in a line, also used as the "left neighbor"
    cost = 0

    # EAP variable: track where to start the next line, and the position of the previous pruning point.
    # Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
    next_start = 0
    prev_pp = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Upper bound: tightened using the last alignment (requires special handling in the code below)
    # Add EPSILON helps dealing with numerical instability
    ub = cutoff + EPSILON - square_cost(lines[nblines - 1], cols[nbcols - 1])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Initialization of the first line
    l0 = lines[0]
    # Fist cell is a special case.
    # Check against the original upper bound dealing with the case where we have both series of length 1.
    cost = square_cost(l0, cols[0])
    if cost > cutoff:
        return np.inf

    buffers[c + 0] = cost
    # All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
    # last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
    curr_pp = 1
    j = 1
    while (j == curr_pp) and (j < nbcols):
        cost = cost + square_cost(l0, cols[j])
        buffers[c + j] = cost
        if cost <= ub:
            curr_pp += 1

        j += 1
    i += 1
    prev_pp = curr_pp

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Main loop
    while i < nblines:
        # --- --- --- Swap and variables init
        swap = c
        c = p
        p = swap

        li = lines[i]
        curr_pp = next_start  # Next pruning point init at the start of the line
        j = next_start

        # --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        cost = buffers[p + j] + square_cost(li, cols[j])
        buffers[c + j] = cost
        if cost <= ub:
            curr_pp = j + 1
        else:
            next_start += 1

        j += 1

        # --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        while (j == next_start) and (j < prev_pp):
            cost = np.minimum(buffers[p + j - 1], buffers[p + j]) + square_cost(li, cols[j])
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            else:
                next_start += 1
            j += 1

        # --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        while j < prev_pp:
            cost = np.minimum(cost, np.minimum(buffers[p + j - 1], buffers[p + j])) + square_cost(li, cols[j])
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            j += 1

        # --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if j < nbcols:  # If so, two cases.
            if j == next_start:  # Case 1: Advancing next start: only diag.
                cost = buffers[p + j - 1] + square_cost(li, cols[j])
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1
                else:
                    # Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                    if (i == nblines - 1) and (j == nbcols - 1) and (cost <= cutoff):
                        return cost
                    else:
                        return np.inf

            else:  # Case 2: Not advancing next start: possible path in previous cells: left and diag.
                cost = np.minimum(cost, buffers[p + j - 1]) + square_cost(li, cols[j])
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1

            j += 1
        else:  # Previous pruning point is out of bound: exit if we extended next start up to here.
            if j == next_start:
                # But only if we are above the original UB
                # Else set the next starting point to the last valid column
                if cost > cutoff:
                    return np.inf
                else:
                    next_start = nbcols - 1

        # --- --- --- Stage 4: After the previous pruning point: only prev.
        # Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        while (j == curr_pp) and (j < nbcols):
            cost = cost + square_cost(li, cols[j])
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp += 1
            j += 1
        # --- --- ---
        prev_pp = curr_pp
        i += 1
    # End of main loop for(;i<nblines;++i)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Finalization
    # Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
    if (j == nbcols) and (cost <= cutoff):
        return cost
    else:
        return np.inf
'''

@njit("float64(float64[:],float64[:],int64,float64)", cache=True, fastmath=True)
def cdtw(lines, cols, w, cutoff):
    # Ensure that lines are longer than columns
    if lines.shape[0] < cols.shape[0]:
        swap = lines
        lines = cols
        cols = swap

    # Cap the windows and check that, given the constraint, an alignment is possible
    if w > lines.shape[0]:
        w = lines.shape[0]

    if (lines.shape[0] - cols.shape[0]) > w:
        return np.inf

    # --- --- --- Declarations
    nblines = lines.shape[0]
    nbcols = cols.shape[0]

    # Setup buffers - get an extra cell for border condition. Init to +INF.
    buffers = np.full(((1 + nbcols) * 2,), np.inf, dtype=np.float64)
    c = 0 + 1  # Start of current line in buffer - account for the extra cell
    p = nbcols + 2  # Start of previous line in buffer - account for two extra cells

    # Line & columns indices
    i = 0
    j = 0

    # Cost accumulator in a line, also used as the "left neighbor"
    cost = 0.0

    # EAP variable: track where to start the next line, and the position of the previous pruning point.
    # Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
    next_start = 0
    prev_pp = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Upper bound: tightened using the last alignment (requires special handling in the code below)
    # Add EPSILON helps dealing with numerical instability
    ub = cutoff + EPSILON - square_cost(lines[nblines - 1], cols[nbcols - 1])

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Initialization of the top border: already initialized to +INF. Initialise the left corner to 0.
    buffers[c - 1] = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Main loop
    for i in range(nblines):
        # --- --- --- Swap and variables init
        swap = c
        c = p
        p = swap

        li = lines[i]
        jStart = np.maximum(i - w, next_start)
        jStop = np.minimum(i + w + 1, nbcols)

        next_start = jStart
        curr_pp = next_start  # Next pruning point init at the start of the line
        j = next_start
        # --- --- --- Stage 0: Initialise the left border
        cost = np.inf
        buffers[c + jStart - 1] = cost

        # --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        while (j == next_start) and (j < prev_pp):
            d = square_cost(li, cols[j])
            cost = np.minimum(buffers[p + j - 1], buffers[p + j]) + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            else:
                next_start += 1
            j += 1

        # --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        while j < prev_pp:
            d = square_cost(li, cols[j])
            cost = np.minimum(cost, np.minimum(buffers[p + j - 1], buffers[p + j])) + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            j += 1

        # --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if j < jStop:  # If so, two cases.
            d = square_cost(li, cols[j])
            if j == next_start:  # Case 1: Advancing next start: only diag.
                cost = buffers[p + j - 1] + d
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1
                else:
                    # Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                    if (i == nblines - 1) and (j == nbcols - 1) and (cost <= cutoff):
                        return cost
                    else:
                        return np.inf

            else:  # Case 2: Not advancing next start: possible path in previous cells: left and diag.
                cost = np.minimum(cost, buffers[p + j - 1]) + d
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1

            j += 1
        else:  # Previous pruning point is out of bound: exit if we extended next start up to here.
            if j == next_start:
                # But only if we are above the original UB
                # Else set the next starting point to the last valid column
                if cost > cutoff:
                    return np.inf
                else:
                    next_start = nbcols - 1

        # --- --- --- Stage 4: After the previous pruning point: only prev.
        # Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        while (j == curr_pp) and (j < jStop):
            d = square_cost(li, cols[j])
            cost = cost + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp += 1
            j += 1

        # --- --- ---
        prev_pp = curr_pp
    # End of main loop for(;i<nblines;++i)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Finalization
    # Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
    if (j == nbcols) and (cost <= cutoff):
        return cost
    else:
        return np.inf


@njit("float64(float64[:],float64[:],float64,float64)", cache=True, fastmath=True)
def ge_dtw(lines, cols, cutoff, ge):
    # Ensure that lines are longer than columns
    if lines.shape[0] < cols.shape[0]:
        swap = lines
        lines = cols
        cols = swap

    # --- --- --- Declarations
    nblines = lines.shape[0]
    nbcols = cols.shape[0]

    # Setup buffers - no extra initialization required - border condition managed in the code.
    buffers = np.zeros((2 * nbcols,), dtype=np.float64)
    c = 0  # Start of current line in buffer - account for the extra cell
    p = nbcols  # Start of previous line in buffer - account for two extra cells

    # Line & columns indices
    i = 0
    j = 0

    # Cost accumulator in a line, also used as the "left neighbor"
    cost = 0

    # EAP variable: track where to start the next line, and the position of the previous pruning point.
    # Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
    next_start = 0
    prev_pp = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Upper bound: tightened using the last alignment (requires special handling in the code below)
    # Add EPSILON helps dealing with numerical instability
    ub = cutoff + EPSILON - cost_function(lines[nblines - 1], cols[nbcols - 1], ge)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Initialization of the first line
    l0 = lines[0]
    # Fist cell is a special case.
    # Check against the original upper bound dealing with the case where we have both series of length 1.
    cost = cost_function(l0, cols[0], ge)
    if cost > cutoff:
        return np.inf

    buffers[c + 0] = cost
    # All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
    # last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
    curr_pp = 1
    j = 1
    while (j == curr_pp) and (j < nbcols):
        cost = cost + cost_function(l0, cols[j], ge)
        buffers[c + j] = cost
        if cost <= ub:
            curr_pp += 1

        j += 1
    i += 1
    prev_pp = curr_pp

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Main loop
    while i < nblines:
        # --- --- --- Swap and variables init
        swap = c
        c = p
        p = swap

        li = lines[i]
        curr_pp = next_start  # Next pruning point init at the start of the line
        j = next_start

        # --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        cost = buffers[p + j] + cost_function(li, cols[j], ge)
        buffers[c + j] = cost
        if cost <= ub:
            curr_pp = j + 1
        else:
            next_start += 1

        j += 1

        # --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        while (j == next_start) and (j < prev_pp):
            cost = np.minimum(buffers[p + j - 1], buffers[p + j]) + cost_function(li, cols[j], ge)
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            else:
                next_start += 1
            j += 1

        # --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        while j < prev_pp:
            cost = np.minimum(cost, np.minimum(buffers[p + j - 1], buffers[p + j])) + cost_function(li, cols[j], ge)
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            j += 1

        # --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if j < nbcols:  # If so, two cases.
            if j == next_start:  # Case 1: Advancing next start: only diag.
                cost = buffers[p + j - 1] + cost_function(li, cols[j], ge)
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1
                else:
                    # Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                    if (i == nblines - 1) and (j == nbcols - 1) and (cost <= cutoff):
                        return cost
                    else:
                        return np.inf

            else:  # Case 2: Not advancing next start: possible path in previous cells: left and diag.
                cost = np.minimum(cost, buffers[p + j - 1]) + cost_function(li, cols[j], ge)
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1

            j += 1
        else:  # Previous pruning point is out of bound: exit if we extended next start up to here.
            if j == next_start:
                # But only if we are above the original UB
                # Else set the next starting point to the last valid column
                if cost > cutoff:
                    return np.inf
                else:
                    next_start = nbcols - 1

        # --- --- --- Stage 4: After the previous pruning point: only prev.
        # Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        while (j == curr_pp) and (j < nbcols):
            cost = cost + cost_function(li, cols[j], ge)
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp += 1
            j += 1
        # --- --- ---
        prev_pp = curr_pp
        i += 1
    # End of main loop for(;i<nblines;++i)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Finalization
    # Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
    if (j == nbcols) and (cost <= cutoff):
        return cost
    else:
        return np.inf


@njit("float64(float64[:],float64[:],int64,float64,float64)", cache=True, fastmath=True)
def ge_cdtw(lines, cols, w, cutoff, ge):
    # Ensure that lines are longer than columns
    if lines.shape[0] < cols.shape[0]:
        swap = lines
        lines = cols
        cols = swap

    # Cap the windows and check that, given the constraint, an alignment is possible
    if w > lines.shape[0]:
        w = lines.shape[0]

    if (lines.shape[0] - cols.shape[0]) > w:
        return np.inf

    # --- --- --- Declarations
    nblines = lines.shape[0]
    nbcols = cols.shape[0]

    # Setup buffers - get an extra cell for border condition. Init to +INF.
    buffers = np.full(((1 + nbcols) * 2,), np.inf, dtype=np.float64)
    c = 0 + 1  # Start of current line in buffer - account for the extra cell
    p = nbcols + 2  # Start of previous line in buffer - account for two extra cells

    # Line & columns indices
    i = 0
    j = 0

    # Cost accumulator in a line, also used as the "left neighbor"
    cost = 0.0

    # EAP variable: track where to start the next line, and the position of the previous pruning point.
    # Must be init to 0: index 0 is the next starting point and also the "previous pruning point"
    next_start = 0
    prev_pp = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Upper bound: tightened using the last alignment (requires special handling in the code below)
    # Add EPSILON helps dealing with numerical instability
    ub = cutoff + EPSILON - cost_function(lines[nblines - 1], cols[nbcols - 1], ge)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Initialization of the top border: already initialized to +INF. Initialise the left corner to 0.
    buffers[c - 1] = 0

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Main loop
    for i in range(nblines):
        # --- --- --- Swap and variables init
        swap = c
        c = p
        p = swap

        li = lines[i]
        jStart = np.maximum(i - w, next_start)
        jStop = np.minimum(i + w + 1, nbcols)

        next_start = jStart
        curr_pp = next_start  # Next pruning point init at the start of the line
        j = next_start
        # --- --- --- Stage 0: Initialise the left border
        cost = np.inf
        buffers[c + jStart - 1] = cost

        # --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        while (j == next_start) and (j < prev_pp):
            d = cost_function(li, cols[j], ge)
            cost = np.minimum(buffers[p + j - 1], buffers[p + j]) + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            else:
                next_start += 1
            j += 1

        # --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        while j < prev_pp:
            d = cost_function(li, cols[j], ge)
            cost = np.minimum(cost, np.minimum(buffers[p + j - 1], buffers[p + j])) + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp = j + 1
            j += 1

        # --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if j < jStop:  # If so, two cases.
            d = cost_function(li, cols[j], ge)
            if j == next_start:  # Case 1: Advancing next start: only diag.
                cost = buffers[p + j - 1] + d
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1
                else:
                    # Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                    if (i == nblines - 1) and (j == nbcols - 1) and (cost <= cutoff):
                        return cost
                    else:
                        return np.inf

            else:  # Case 2: Not advancing next start: possible path in previous cells: left and diag.
                cost = np.minimum(cost, buffers[p + j - 1]) + d
                buffers[c + j] = cost
                if cost <= ub:
                    curr_pp = j + 1

            j += 1
        else:  # Previous pruning point is out of bound: exit if we extended next start up to here.
            if j == next_start:
                # But only if we are above the original UB
                # Else set the next starting point to the last valid column
                if cost > cutoff:
                    return np.inf
                else:
                    next_start = nbcols - 1

        # --- --- --- Stage 4: After the previous pruning point: only prev.
        # Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        while (j == curr_pp) and (j < jStop):
            d = cost_function(li, cols[j], ge)
            cost = cost + d
            buffers[c + j] = cost
            if cost <= ub:
                curr_pp += 1
            j += 1

        # --- --- ---
        prev_pp = curr_pp
    # End of main loop for(;i<nblines;++i)

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # Finalization
    # Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
    if (j == nbcols) and (cost <= cutoff):
        return cost
    else:
        return np.inf
