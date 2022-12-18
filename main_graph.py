import time
import torch
import os

from torch_geometric.utils import dropout_adj
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from dataloader import load_dataset
from evaluation import eval_graph
from utils import *
from model import WGDN


def loss_func(feat, recons, mask=None, ratio=0):
    
    # Reconstruction loss
    residual = (recons- feat).pow(2)
    del feat, recons
    
    if mask is not None and ratio != 0:
        residual[mask[0], mask[1]] = ratio * residual[mask[0], mask[1]]

    cost = residual.mean(1).sqrt().mean()

    return cost


def train(model, train_dataloader, eval_loader, logger, args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.decay_rate)

    eval_acc = []

    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
        
        avg_loss = 0
        epoch_time = 0

        for step, batch in enumerate(train_dataloader):
            batch = batch.to(args.device)
            
            # Laplacian matrix
            edge_index_drop, edge_weight_drop = get_normalization(dropout_adj(batch.edge_index, num_nodes=batch.x.size(0), p=args.edge_dropout)[0], 
                                                                            batch.x.size(0), False, batch.x.dtype, args.norm_type)
            avg_edge_index, avg_edge_weight = get_normalization(batch.edge_index, batch.x.size(0), False, batch.x.dtype, 'rw', laplacian=False)

            if args.sparse:
                lap_adj = SparseTensor(row=edge_index_drop[0], col=edge_index_drop[1], value=edge_weight_drop, 
                                sparse_sizes=(batch.x.size(0), batch.x.size(0)))
                avg_adj = SparseTensor(row=avg_edge_index[0], col=avg_edge_index[1], value=avg_edge_weight, 
                                        sparse_sizes=(batch.x.size(0), batch.x.size(0)))
            
            nnz_mask = (batch.x != 0)
            balance_ratio = (~nnz_mask).sum()/nnz_mask.sum()
            nnz_mask = torch.where(nnz_mask)
            
            model.train()
            start_time = time.time()

            if not args.sparse:
                node_rec = model(batch.x, edge_index_drop, edge_weight_drop, avg_edge_index, avg_edge_weight)
            else:
                node_rec = model(batch.x, lap_adj, avg_adj)
            loss = loss_func(batch.x, node_rec, nnz_mask, balance_ratio)

            optimizer.zero_grad()
            loss.backward()

            # In case nan occurs
            loop = 0
            while list(model.parameters())[3].grad.isnan().any() and loop < 5:
                if not args.sparse:
                    node_rec = model(batch.x, edge_index_drop, edge_weight_drop, avg_edge_index, avg_edge_weight)
                else:
                    node_rec = model(batch.x, lap_adj, avg_adj)
                loss = loss_func(batch.x, node_rec, nnz_mask, balance_ratio)

                optimizer.zero_grad()
                loss.backward()

                loop += 1

            optimizer.step()

            avg_loss += loss.item()
            end_time = time.time()
            epoch_time += end_time - start_time

        avg_loss /= (step + 1)
        scheduler.step()

        epoch_iter.set_description(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.eval_epoch == 0 and not args.final and (epoch + 1) != args.epoch:
            model.eval()
            test_acc, test_std = eval_graph(model, eval_loader, args)
            eval_acc.append(test_acc)
            logger.info('Seed: {}, Epoch:{},  Mean accuracy: {}, Standard deviation: {}'.format(seed, epoch + 1, test_acc, test_std))
    
    return model, eval_acc


if __name__ == '__main__':

    # Configurations
    args = build_args('graph')
    if args.use_cfg:
        args = load_best_configs(args, "config.yml")
    print_config(vars(args))

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    # RSNR parameter
    if not args.single:
        args.gamma = args.gamma_list
    else:
        args.gamma = [args.gamma]

    # Create logger
    logger, model_info = create_logger(args, 'graph')
    
    # Model preparation
    dataset = load_dataset(args.data_dir, args.dataset)
    args.n_classes = dataset.data.y.max().item() + 1

    # Data preprocessing
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Modeling
    acc_list = []
    eval_acc_list = []
    for seed in args.seed:
        set_random_seed(seed)

        gnn_model = WGDN(dataset.data.x.size(1), args.hid_dim, args.num_layer, args.gamma, kernel=args.kernel, 
                    skip=(not args.nonskip), drop_ratio=args.dropout, act=args.act, norm=args.norm, 
                    dec_aggr=args.dec_aggr, large=args.large, beta=args.beta, emb_act=True).to(args.device)

        model_file = model_info + '_seed_{}'.format(str(seed))
        args.model_path = os.path.join(args.model_dir, args.dataset.lower(), model_file + '.pkl')

        if not args.load_model:
            gnn_model, eval_acc = train(gnn_model, train_loader, eval_loader, logger, args)
            gnn_model.eval()
            logger.info("Training finished, activation function: {}".format(args.act))

            if not args.final:
                eval_acc_list.append(eval_acc)
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
        logger.info('Graph pooling mechanism: {}, batch size: {}'.format(args.pooler, args.batch_size))
        test_acc, test_std = eval_graph(gnn_model, eval_loader, args)

        logger.info('Seed: {}, Mean accuracy: {}, Standard deviation: {}'.format(seed, test_acc, test_std))
        logger.info('')

        acc_list.append(test_acc)
    
    # Final evaluation
    if len(args.seed) > 1:

        if not args.load_model and not args.final:
            eval_acc_list = np.array(eval_acc_list)
            for i in range(eval_acc_list.shape[1]):
                eval_acc_avg, eval_acc_std = np.mean(eval_acc_list[:, i]), np.std(eval_acc_list[:, i])
                epoch = (i + 1) * args.eval_epoch
                logger.info('Epoch:{},  accuracy: {}, Standard deviation: {}'.format(epoch, eval_acc_avg, eval_acc_std))

        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        logger.info('Final accuracy: {}, Standard deviation: {}'.format(final_acc, final_acc_std))
        logger.info('')