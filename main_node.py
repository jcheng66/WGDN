import time
import torch
import os

from torch_geometric.utils import dropout_adj
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloader import load_dataset
from evaluation import eval_node
from utils import *
from model import WGDN

from torch.nn import MSELoss


def loss_func(feat, recons, mask=None, ratio=0):
    
    # Reconstruction loss
    residual = (recons - feat).pow(2)
    del feat, recons
    
    if mask is not None and ratio != 0:
        residual[mask[0], mask[1]] = ratio * residual[mask[0], mask[1]]

    cost = residual.sum(1).mean().sqrt()

    return cost


def train(model, data, logger, args):

    # Preprocessing
    edge_index, edge_weight = get_normalization(data.edge_index, data.x.size(0), False, data.x.dtype, args.norm_type)
    avg_edge_index, avg_edge_weight = get_normalization(data.edge_index, data.x.size(0), False, data.x.dtype, 'rw', laplacian=False)
    if not args.sparse:
        edge_index = edge_index.to(args.device)
        edge_weight = edge_weight.to(args.device)
        avg_edge_index = avg_edge_index.to(args.device)
        avg_edge_weight = avg_edge_weight.to(args.device)
    else:
        lap_adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                                sparse_sizes=(data.x.size(0), data.x.size(0))).to(args.device)
        avg_adj = SparseTensor(row=avg_edge_index[0], col=avg_edge_index[1], value=avg_edge_weight, 
                                sparse_sizes=(data.x.size(0), data.x.size(0))).to(args.device)


    features = data.x.to(args.device)

    if args.dataset in ['WikiCS', 'ogbn-arxiv']:
        nnz_mask = None
        balance_ratio = 1
    else:
        nnz_mask = (features != 0)
        balance_ratio = (~nnz_mask).sum()/nnz_mask.sum()
        nnz_mask = torch.where(nnz_mask)

    logger.info(f'Activation function: {args.act}')
    if args.single:
        logger.info(f'Kernel: {args.kernel}, Emb Size: {args.hid_dim}, Num Layer: {args.num_layer}, Self-supervised settings')
    else:
        logger.info(f'Kernel: {args.kernel}, Emb Size: {args.hid_dim}, Num Layer: {args.num_layer}, Aggregation: {args.dec_aggr}, Self-supervised settings')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    criterion = MSELoss()

    # Reconstruction-based learning
    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
        
        # Edge dropout
        if args.edge_dropout > 0:
            edge_index_drop, edge_weight_drop = get_normalization(dropout_adj(data.edge_index, num_nodes=features.size(0), p=args.edge_dropout)[0], 
                                                                    data.x.size(0), False, features.dtype, args.norm_type)
            edge_index_drop = edge_index_drop.to(args.device)
            edge_weight_drop = edge_weight_drop.to(args.device)
        else:
            edge_index_drop, edge_weight_drop = edge_index, edge_weight

        # Recording
        epoch_time = 0
        model.train()

        optimizer.zero_grad()
        start_time = time.time()

        if not args.sparse:
            node_rec = model(features, edge_index_drop, edge_weight_drop, avg_edge_index, avg_edge_weight)
        else:
            node_rec = model(features, lap_adj, avg_adj)
        
        if args.dataset in ['ogbn-arxiv']:
            loss = criterion(node_rec, features)
        else:
            loss = loss_func(features, node_rec, nnz_mask, balance_ratio)

        loss.backward()
        optimizer.step()
        scheduler.step()

        end_time = time.time()
        epoch_time += end_time - start_time

        epoch_iter.set_description(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

        if (epoch + 1) % args.eval_epoch == 0 and not args.final and (epoch + 1) != args.epoch:
            model.eval()
            if not args.sparse:
                node_rep = model.get_embedding(features, edge_index, edge_weight)
            else:
                node_rep = model.get_embedding(features, lap_adj)

            eval_node(data, node_rep, logger, args, final=False, epoch=epoch + 1)

    return model


if __name__ == '__main__':

    # Configurations
    args = build_args('node')
    if args.use_cfg:
        args = load_best_configs(args, "config.yml")
    print_config(vars(args))

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    # AER parameter
    if not args.single:
        args.gamma = args.gamma_list
    else:
        args.gamma = [args.gamma]
    
    # Create logger
    logger, model_info = create_logger(args, 'node')

    # Model preparation
    data = load_dataset(args.data_dir, args.dataset)
    args.n_classes = data.y.max().item() + 1

    set_random_seed(args.seed[0])

    gnn_model = WGDN(data.x.size(1), args.hid_dim, args.num_layer, args.gamma, kernel=args.kernel, 
                    skip=(not args.nonskip), drop_ratio=args.dropout, norm=args.norm, act=args.act, 
                    dec_aggr=args.dec_aggr, large=args.large, beta=args.beta, emb_act=args.emb_act, 
                    simple=args.simple).to(args.device)

    # Model directory
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, args.dataset.lower()), exist_ok=True)
    args.model_path = os.path.join(args.model_dir, args.dataset.lower(), model_info + '.pkl')

    if not args.load_model:
        gnn_model = train(gnn_model, data, logger, args)
        gnn_model.eval()
        logger.info("Training finished, activation function: {}".format(args.act))
    else:
        if not os.path.exists(args.model_path):
            raise ValueError("Pre-trained model does not exist! Please pre-train before evaluation!")
        else:    
            logger.info("Restore pre-trained model, activation function: {}".format(args.act))
            gnn_model.load_state_dict(torch.load(args.model_path))
            gnn_model.eval()

    if args.save_model:
        logger.info("Saving model")
        torch.save(gnn_model.state_dict(), args.model_path)               

    logger.info('Downstream task evaluation starts!')
    logger.info('Node classification L2 penalty: {}'.format(args.down_l2))


    edge_index, edge_weight = get_normalization(data.edge_index, data.x.size(0), False, data.x.dtype, args.norm_type)
    features = data.x.to(args.device)

    if not args.sparse:
        edge_index = edge_index.to(args.device)
        edge_weight = edge_weight.to(args.device)

        node_rep = gnn_model.get_embedding(features, edge_index, edge_weight)
    else:
        lap_adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                            sparse_sizes=(data.x.size(0), data.x.size(0))).to(args.device)

        node_rep = gnn_model.get_embedding(features, lap_adj)
    
    eval_node(data, node_rep, logger, args)