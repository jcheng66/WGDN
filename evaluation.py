import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_sparse import SparseTensor

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import get_normalization
from model import LinearPred


def eval_node(data, node_rep, logger, args, final=True, epoch=None):

    accs = []
    node_rep = node_rep.detach()
    labels = data.y.to(args.device)

    for k in range(args.down_run):
        
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv']:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else: 
            train_mask = data.train_mask[k]
            val_mask = data.val_mask[k]
            test_mask = data.test_mask [k] if args.dataset != 'WikiCS' else data.test_mask


        pred_model = LinearPred(node_rep.size(1), args.n_classes).to(args.device)
        optimizer_pred = torch.optim.Adam(pred_model.parameters(), lr=args.down_lr, weight_decay=args.down_l2)

        acc_val_best, acc_test_best, ep_b = 0, 0, -1
        acc_test_best_all = 0
        for j in range(args.down_epoch):
            pred_model.train()
            optimizer_pred.zero_grad()

            logits = pred_model(node_rep[train_mask])
            loss = F.cross_entropy(logits, labels[train_mask])

            loss.backward()
            optimizer_pred.step()

            pred_model.eval()
            preds_val = torch.argmax(pred_model(node_rep[val_mask]), dim=1)
            preds_test = torch.argmax(pred_model(node_rep[test_mask]), dim=1)
            acc_val = torch.sum(preds_val == labels[val_mask]).float() / labels[val_mask].size(0)
            acc_test = torch.sum(preds_test == labels[test_mask]).float() / labels[test_mask].size(0)

            if acc_val >= acc_val_best:
                acc_val_best = acc_val
                if acc_test >= acc_test_best:
                    acc_test_best = acc_test
                    ep_b = j
            
            if acc_test > acc_test_best_all:
                acc_test_best_all = acc_test
        
        accs.append(acc_test_best * 100)
        if final:    
            logger.info(f'Epoch: {k}, Accuracy: {acc_test_best} from {ep_b} step, Best Accuracy: {acc_test_best_all}')
    
    accs = torch.stack(accs)
    if final:
        logger.info('Final Mean accuracy: {}, Standard deviation: {}'.format(accs.mean().item(), accs.std().item()))
        logger.info('')
    else:
        logger.info('Epoch:{}, Mean accuracy: {}, Standard deviation: {}'.format(epoch, accs.mean().item(), accs.std().item()))


def eval_graph(model, dataloader, args):

    # Pooling function
    if args.pooler == 'avg':
        pooler = global_mean_pool
    elif args.pooler == 'sum':
        pooler = global_add_pool
    elif args.pooler == 'max':
        pooler = global_max_pool

    # Obtain graph embeddings
    embed_list = []
    y_list = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            
            # Laplacian matrix
            edge_index, edge_weight = get_normalization(batch.edge_index, batch.x.size(0), False, batch.x.dtype, args.norm_type)
            if args.sparse:
                lap_adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                                sparse_sizes=(batch.x.size(0), batch.x.size(0)))
                
            if not args.sparse:
                node_rep = model.get_embedding(batch.x, edge_index, edge_weight)
            else:
                node_rep = model.get_embedding(batch.x, lap_adj)
            graph_rep = pooler(node_rep, batch.batch)

            embed_list.append(graph_rep.cpu().numpy())
            y_list.append(batch.y.cpu().numpy())
    embed = np.concatenate(embed_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_acc, test_std = evaluate_graph_embeddings_using_svm(embed, y)
    
    return test_acc, test_std


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params, cv=5, scoring='accuracy', verbose=0)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        result.append(acc)
    test_acc = np.mean(result)
    test_std = np.std(result)

    return test_acc, test_std
