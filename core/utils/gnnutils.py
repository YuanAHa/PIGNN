import os.path as osp
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_add


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def get_adj(edge_index, weight=None):
    """return adjacency matrix"""
    # edge_index_ = torch.stack(edge_index, dim=0)
    if not weight:
        weight = torch.ones(edge_index.shape[1], device=edge_index[0].device)

    row, col = edge_index
    return torch.sparse.FloatTensor(edge_index, weight)


def get_laplacian(edge_index_, weight=None, type='norm', sparse=True):
    """return Laplacian (sparse tensor)
    type: 'comb' or 'norm' for combinatorial or normalized one.
    """
    if not isinstance(edge_index_, torch.Tensor):
        edge_index = torch.stack(edge_index_, dim=0)
    else:
        edge_index = edge_index_
    adj = get_adj(edge_index, weight=weight)    # torch.sparse.FloatTensor
    num_nodes = adj.shape[1]
    senders, receivers = edge_index
    num_edges = edge_index.shape[1]
    
    deg = scatter_add(torch.ones(num_edges, device=senders.device), senders)
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)], device=deg.device), deg)
    Laplacian = sp_deg - adj    # L = D-A
    
    deg = deg.pow(-0.5)
    deg[deg == float('inf')] = 0
    sp_deg = torch.sparse.FloatTensor(torch.tensor([range(num_nodes),range(num_nodes)], device=deg.device), deg)
    Laplacian_norm = sp_deg.mm(Laplacian.mm(sp_deg.to_dense()))     # Lsym = (D^-1/2)L(D^-1/2)
    
    if type=="comb":
        return Laplacian if sparse else Laplacian.to_dense()
    elif type=="norm":
        return to_sparse(Laplacian_norm) if sparse else Laplacian_norm
    else:
        raise ValueError("type should be one of ['comb', 'norm']")


def decompose_graph(graph):

    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    graph_info = {k:graph[k] for k in graph.keys}
    ret = Data(**graph_info)
    
    return ret
