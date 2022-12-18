import os
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, TUDataset, WikiCS
from torch_geometric.utils import degree, to_undirected
from sklearn.preprocessing import StandardScaler


# ======================================================================
#   Data loader core functions
# ======================================================================

def load_dataset(root, dataset):
    """
    Wrapper functions for data loader
    """

    if dataset in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']:
        dataset = load_node_dataset(root, dataset)
    elif dataset == 'WikiCS':
        dataset = WikiCS(os.path.join(root, dataset.lower()))[0]
        dataset.train_mask = dataset.train_mask.T
        dataset.val_mask = dataset.val_mask.T
    elif dataset in ['IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'COLLAB', 'MUTAG', 'REDDIT-BINARY', 'NCI1', 'DD', 'PTC_MR']:
        dataset = load_tu_dataset(root, dataset)
    elif dataset in ['ogbn-proteins', 'ogbn-arxiv']:
        dataset = load_ogbn_dataset(dataset)
        
    return dataset


def load_node_dataset(root, dataset):
    """
    Data loader for Node Classification Dataset
    """
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root, dataset)
    elif dataset in ['CS', 'Physics']:
        dataset = Coauthor(root, dataset)
    elif dataset in ['Computers', 'Photo']:
        dataset = Amazon(root, dataset)

    name = dataset.name
    dataset = dataset[0]
    if name.lower() in ['computers', 'photo', 'cs', 'physics']:
        dataset = train_val_test_split(dataset, 20)
    
    return dataset


def load_tu_dataset(root, dataset):
    '''
    Data loader for TUDataset
    '''
    dataset = TUDataset(root, dataset)
    
    if dataset.num_node_labels > 0 and not dataset.name in ['MUTAG']:
        print('Use node labels as attributes.')
    else:
        print('Use node degrees as attributes.')

        if not hasattr(dataset.data, '_num_nodes'):
            dataset.data._num_nodes = dataset.slices['x'].diff()

        # Calculate node degrees
        feat_dim, MAX_DEGREE = 0, 400
        degrees = []
        x_slices = [0]
        for i in range(dataset.len()):
            start, end = dataset.slices['edge_index'][i:(i+2)].tolist()
            num_nodes = dataset.data._num_nodes[i]
            cur_degree = degree(dataset.data.edge_index[0, start:end], num_nodes).long()
            feat_dim = max(feat_dim, cur_degree.max().item())
            degrees.append(cur_degree)
            x_slices.append(x_slices[-1] + num_nodes)

        degrees = torch.cat(degrees)

        # Restrict maximum degree
        feat_dim = min(feat_dim, MAX_DEGREE) + 1
        degrees[degrees > MAX_DEGREE] = MAX_DEGREE
        x = F.one_hot(degrees, num_classes=feat_dim).float()

        # Record
        dataset.data.x = x
        dataset.slices['x'] = torch.tensor(x_slices, dtype=dataset.slices['edge_index'].dtype)

    return dataset


def load_ogbn_dataset(dataset):
    """
    Dataloader for OGBN Dataset
    """
    dataset = PygNodePropPredDataset(dataset)

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    dataset = dataset[0]
    num_nodes = dataset.x.size(0)
    dataset.y = dataset.y.view(-1)

    scaler = StandardScaler()
    dataset.x = torch.from_numpy(scaler.fit_transform(dataset.x.numpy()))
    dataset.edge_index = to_undirected(dataset.edge_index)

    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)

    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask

    return dataset


def train_val_test_split(data, run, train_ratio=0.1, val_ratio=0.1):
    """
    Train-Test split for node classification data (Aligning with baseline split)
    """
    num_nodes = data.y.shape[0]

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    for i in range(run):

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        torch.manual_seed(i)
        shuffle_idx = torch.randperm(num_nodes)
        train_idx = shuffle_idx[:num_train]
        val_idx = shuffle_idx[num_train:(num_train + num_val)]
        test_idx = shuffle_idx[(num_train + num_val):]
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_mask = train_mask.reshape(1, -1)
        val_mask = val_mask.reshape(1, -1)
        test_mask = test_mask.reshape(1, -1)

        if i == 0:
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        else:
            data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
            data.val_mask = torch.cat((data.val_mask, val_mask), dim=0)
            data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    return data