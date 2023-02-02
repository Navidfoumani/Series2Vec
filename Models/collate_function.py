import torch
import numpy as np
import tslearn

EPSILON = 10e-12


def collate_fn(data):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    batch_size = len(data)
    features, labels, IDs = zip(*data)
    X = torch.zeros(batch_size, features[0].shape[0], features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    Dtw_Dist = torch.tensor(tslearn.metrics.cdist_dtw(features))

    '''
    for i in range(batch_size):
        X[i] = features[i]
        for j in range(i+1, batch_size):
            Dtw_Dist[i, j] = tslearn.metrics.dtw(features[i], features[j])
            Dtw_Dist[j, i] = Dtw_Dist[i, j]
    
    for i in range(batch_size):
        X[i] = features[i]
        for j in range(i+1, batch_size):
            for k in range(features[i].shape[0]):
                Sim[i, j] += cdtw(features[i][k], features[j][k], round(0.1 * features[i].shape[-1]), np.inf)
            Sim[j, i] = Sim[i, j]
    '''
    return X, Dtw_Dist, torch.tensor(labels), IDs


'''
for i in range(batch_size):
    X[i] = features[i]
    for j in range(batch_size):
        for k in range(features[i].shape[0]):
            tem = 0
            tem = cdtw(features[i][k], features[j][k], round(0.1*features[i].shape[-1]), np.inf)
            Sim[i, j] += tem
        Sim[j, i] = Sim[i, j]
        
'''


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


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def DataTransform(sample):
    weak_aug = scaling(sample, 1.1)
    strong_aug = jitter(permutation(sample, max_segments=8), 1.1)

    return torch.tensor(weak_aug, dtype=torch.float32), torch.tensor(strong_aug, dtype=torch.float32)


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


def DataMasking(x):
    dropout = torch.nn.Dropout(0.25)
    ai = []
    for i in range(x.shape[0]):
        xi = dropout(x[i, :, :])
        x[i] = xi
    return x
