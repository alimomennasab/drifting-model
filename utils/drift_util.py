def computeV(x, y_pos, y_neg, T):
    # x: [N, D]
    # y_pos: [N_pos, D]
    # y_neg: [N_neg, D]
    # T: temperature

    # compute pairwise distance
    dist_pos = cdist(x, y_pos) # [N, N_pos]
    dist_neg = cdist(x, y_neg) # [N, N_neg]

    # ignore self (if y_neg is x)
    dist_neg += eye(N) * 1e6

    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T

    # concat for normalization
    logit = cat([logit_pos, logit_neg], dim=1)

    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = sqrt(A_row * A_col)

    # back to [N, N_pos] and [N, N_neg]
    A_pos, A_neg = split(A, [N_pos,], dim=1)

    # compute the weights
    W_pos = A_pos # [N, N_pos]
    W_neg = A_neg # [N, N_neg]
    W_pos *= A_neg.sum(dim=1,keepdim=True)
    W_neg *= A_pos.sum(dim=1,keepdim=True)

    drift_pos = W_pos @ y_pos # [N_x, D]
    drift_neg = W_neg @ y_neg # [N_x, D]
    
    V = drift_pos - drift_neg

    return V

