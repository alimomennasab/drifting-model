import torch    

def compute_attraction(y, real_data, tau):
    # y: one-step sample
    # real_data: real data batch
    # tau: temperature
    squared_distances = torch.cdist(y, real_data).square()
    weights = torch.softmax(-squared_distances / (2 * tau ** 2), dim=-1)
    attraction = weights @ real_data

    return attraction


def compute_repulsion(y, sigma_r=1.5):
    # y: one-step sample

    squared_distances = torch.cdist(y, y).square()
    weights = torch.exp(-squared_distances / (2 * sigma_r ** 2))
    self_mask = torch.eye(
        y.shape[0], dtype=torch.bool, device=y.device
    )
    weights = weights.masked_fill(self_mask, 0)

    weight_sum = weights.sum(dim=-1, keepdim=True)
    weighted_neighbors = weights @ y

    repulsion = (weight_sum * y - weighted_neighbors) / weight_sum.clamp_min(1e-8)

    return repulsion


def compute_sharpener_drift(
    y,
    real_data,
    tau,
    lambda_rep=0.1,
    sigma_r=1.5,
):
    # y: one-step sample
    # real_data: real (positive) data sample
    # tau: temperature

    attraction = compute_attraction(y, real_data, tau)
    repulsion = compute_repulsion(y, sigma_r)
    v = (attraction - y) - lambda_rep * repulsion
    return v


def computeV(x, y_pos, y_neg, T):
    # x: [HW, B, C] 
    # y_pos: [HW, N_pos, C]
    # y_neg: [HW, N_neg, C]
    # T: temperature
    num_x = x.shape[-2]
    num_pos = y_pos.shape[-2]
    num_neg = y_neg.shape[-2]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # [HW, B, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [HW, B, N_neg]

    # ignore self in distance computation (if y_neg is x)
    if y_neg is x:
        if num_x != num_neg:
            raise ValueError("Self-masking requires equal x and y_neg batch sizes.")

        # - we create a boolean identity matrix of size [B, B]
        # [[True,  False, False],
        # [False, True,  False],
        # [False, False, True ]]
        # - we add a dim -> [1, B, B], so we can broadcast to every distance in dist_neg
        # - we make all true entries infinity. these true entries are each sample's distance with itself
        # - later, logit_neg = -dist_neg / T makes all inf -> -inf
        # - then, when softmaxed, these -inf's become 0
        # - so, each sample's distance with itself is not utilized in the V calculation 

        self_mask = torch.eye(
            num_x, dtype=torch.bool, device=x.device
        ).unsqueeze(0)
        dist_neg = dist_neg.masked_fill(self_mask, float("inf"))

    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T

    # concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=-1)

    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = torch.sqrt(A_row * A_col)

    # back to [HW, B, N_pos] and [HW, B, N_neg]
    A_pos, A_neg = torch.split(A, [num_pos, num_neg], dim=-1)

    # compute the weights
    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    drift_pos = W_pos @ y_pos  # [HW, B, C]
    drift_neg = W_neg @ y_neg  # [HW, B, C]
    
    V = drift_pos - drift_neg

    return V

