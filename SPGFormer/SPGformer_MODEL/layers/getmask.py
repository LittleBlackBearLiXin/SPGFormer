import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_seed(42)


@torch.no_grad()
def get_mask(net_input, device, num_subgraphs=0, token=50,row=True):
    height, width, _ = net_input.shape

    if row:
        out = net_input.reshape([height * width, -1]).to(device)
    else:
        out = net_input.permute(1, 0, 2).reshape(width * height, -1)
    out = F.layer_norm(out, [out.size(-1)])
    N = height * width


    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )

    coordinates = torch.stack((rows.flatten(), cols.flatten()), dim=1)



    mask_src, mask_dst = [], []


    subgraph_size = N // num_subgraphs
    subgraph_ranges = [(i * subgraph_size, min((i + 1) * subgraph_size, N))
                       for i in range(num_subgraphs)]


    for i in range(num_subgraphs):
        curr_start, curr_end = subgraph_ranges[i]
        curr_indices = torch.arange(curr_start, curr_end, device=device)
        curr_out = out[curr_indices]
        curr_coords = coordinates[curr_indices]


        if curr_end - curr_start > 1:
            feat_sim = torch.mm(curr_out, curr_out.T)
            coord_dist = torch.cdist(curr_coords.float(), curr_coords.float(), p=2)**2
            spatial_sim = torch.exp(-coord_dist / 4.0)
            sim_matrix = torch.relu(feat_sim) * spatial_sim

            k = min(token, curr_end - curr_start - 1)
            topk_val, topk_idx = torch.topk(sim_matrix, k=k, dim=1)
            #print(topk_idx)
            #print(curr_indices)

            src = curr_indices.view(-1, 1).expand(-1, k)
            #print(src)
            dst = curr_indices[topk_idx]
            #print(dst)

            mask_src.append(src.flatten())
            mask_dst.append(dst.flatten())


    if mask_src:
        sparse_mask = torch.sparse_coo_tensor(
            torch.stack([torch.cat(mask_src), torch.cat(mask_dst)]),
            torch.ones(sum(len(es) for es in mask_src), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).coalesce().to_sparse_csr()
    else:
        sparse_mask = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float, device=device),
            size=(N, N),
            device=device
        ).to_sparse_csr()

    return sparse_mask


from torch_geometric.utils import remove_self_loops, add_self_loops

def row_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 2
        token = 30
    elif FLAG==2:
        num_subgraphs = 40
        token = 20
    elif FLAG==3:
        num_subgraphs = 4
        token = 25
    else:
        num_subgraphs = 0
        token = 0

    return num_subgraphs, token

def col_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 2
        token = 30

    elif FLAG==2:
        num_subgraphs = 20
        token = 20

    elif FLAG==3:
        num_subgraphs = 4
        token = 25
    else:
        num_subgraphs = 0
        token = 0

    return num_subgraphs, token

from torch_scatter import scatter_add

def simfunction(mask):
    mask=mask.to_sparse_coo()
    mask_index = mask.indices()
    mask_attr = mask.values()

    mask_index, mask_attr = remove_self_loops(mask_index, mask_attr)
    mask_index, mask_attr = add_self_loops(mask_index, mask_attr, num_nodes=mask.size(0))

    src, dst = mask_index[0], mask_index[1]
    row_sum = scatter_add((mask_attr), src, dim_size=mask.shape[0])**0.5
    col_sum = scatter_add((mask_attr), dst, dim_size=mask.shape[0]) ** 0.5
    att = (row_sum[src]) * mask_attr * (col_sum[dst])

    mask = torch.sparse_coo_tensor(
        mask_index,
        att,
        size=mask.size(),
        device=mask.device
    ).coalesce()

    return mask




def process_mask(net_input, device,FLAG=-1):
    row_num_subgraphs,row_top_k=row_hyp(FLAG)
    col_num_subgraphs,col_top_k=col_hyp(FLAG)

    row_mask = get_mask(net_input, device, num_subgraphs=row_num_subgraphs, token=row_top_k,row=True)
    col_mask = get_mask(net_input, device, num_subgraphs=col_num_subgraphs, token=col_top_k, row=False)
    rowmask = simfunction(row_mask)
    colmask = simfunction(col_mask)

    return rowmask,colmask










def PSPT_row_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 1
        token = 120
    elif FLAG==2:
        num_subgraphs = 40
        token = 20
    elif FLAG==3:
        num_subgraphs = 4
        token = 25
    else:
        num_subgraphs = 0
        token = 0

    return num_subgraphs, token

def PSPT_col_hyp(FLAG):
    if FLAG==1:
        num_subgraphs = 2
        token = 30

    elif FLAG==2:
        num_subgraphs = 20
        token = 20

    elif FLAG==3:
        num_subgraphs = 4
        token = 20
    else:
        num_subgraphs = 0
        token = 0

    return num_subgraphs, token



def PSPT_process_mask(net_input, device,FLAG=-1):
    row_num_subgraphs,row_top_k=PSPT_row_hyp(FLAG)
    col_num_subgraphs,col_top_k=PSPT_col_hyp(FLAG)

    row_mask = get_mask(net_input, device, num_subgraphs=row_num_subgraphs, token=row_top_k,row=True)
    col_mask = get_mask(net_input, device, num_subgraphs=col_num_subgraphs, token=col_top_k, row=False)
    rowmask = simfunction(row_mask)
    colmask = simfunction(col_mask)

    return rowmask,colmask