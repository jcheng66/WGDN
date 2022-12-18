import os
import yaml
import random
import logging
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np

from functools import partial

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from numpy.polynomial import Polynomial, chebyshev


# ======================================================================
#   Reproducibility
# ======================================================================

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.determinstic = True


# ======================================================================
#   Configuration functions
# ======================================================================

def build_args(task):

    parser = argparse.ArgumentParser(description='WGDN')
    # General settings
    parser.add_argument("--model", type=str, default="WGDN", help="Model type")
    parser.add_argument("--kernel", type=str, default="heat", help="Encoding kernel type")
    parser.add_argument("--dataset", type=str, default="PubMed", help="Dataset for this model")
    parser.add_argument("--data_dir", type=str, default="./dataset/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="./model/", help="Folder to save model")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Folder to save logger")

    # Model Configuration settings
    parser.add_argument("--seed", type=int, nargs="+", default=[12], help="Random seed")
    parser.add_argument("--hid_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of hidden layer in main model")
    parser.add_argument("--act", type=str, default='prelu', help="Activation function type")
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    
    parser.add_argument("--dec_aggr", type=str, default="max", help="Aggregation for multi-chanel decoder")
    parser.add_argument("--single", action='store_true', default=False, help="Indicator of single channel")
    parser.add_argument("--large", action='store_true', default=False, help="Indicator of two layer deconvolution in decoder")
    parser.add_argument("--nonskip", action="store_true", default=False, help="Indicator of non-skip connection")
    parser.add_argument("--simple", action="store_true", default=False, help="Indicator of 1 layer decoder")

    # Training settings
    parser.add_argument("--epoch", type=int, default=300, help="The max number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of optimizer")
    parser.add_argument("--l2", type=float, default=0, help="Coefficient of L2 penalty")
    parser.add_argument("--decay_rate", type=float, default=0.98, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument("--eval_epoch", type=int, default=100, help="Number of evaluation epoch")
    parser.add_argument("--sparse", action='store_true', default=False, help="Indicator of sparse computation")
    
    # Hyperparameters
    parser.add_argument("--norm_type", type=str, default='sym', help="Type of normalization of adjacency matrix")
    parser.add_argument("--beta", type=float, default=1, help="Hyperparameter for latent augmentation")
    parser.add_argument("--gamma", type=float, default=1, help="Hyperparameter for AER")
    parser.add_argument("--gamma_list", type=float, default=[0.1, 1, 10], nargs="+", help="Hyperparameter for AER list")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for node in training")
    parser.add_argument("--edge_dropout", type=float, default=0, help="Dropout rate for edge in training")
    
    # Auxiliary
    parser.add_argument("--save_model", action='store_true', default=False, help="Indicator to save trained model")
    parser.add_argument("--load_model", action='store_true', default=False, help="Indicator to load trained model")
    parser.add_argument("--final", action='store_true', default=False, help="Indicator to not evaluate intermediate results")
    parser.add_argument("--log", action='store_true', default=False, help="Indicator to write logger file")
    parser.add_argument("--use_cfg", action="store_true", default=False, help="Indicator to use best configurations")
    
    # GPU settings
    parser.add_argument("--no_cuda", action='store_true', default=False, help="Indicator of GPU availability")
    parser.add_argument("--device", type=int, default=0, help='Which gpu to use if any')

    # Task-related settings
    if task == 'node':
        parser.add_argument("--emb_act", action="store_true", default=False, help="Indicator of using activation for node embedding")

        parser.add_argument("--down_epoch", type=int, default=300, help="The max number of epochs in downstream tasks")
        parser.add_argument("--down_lr", type=float, default=0.01, help="Learning rate of optimizer of downstream tasks")
        parser.add_argument("--down_run", type=int, default=20, help="Number of evaluation in downstream tasks")
        parser.add_argument("--down_decay_rate", type=float, default=1, help="Decay rate of learning rate in downstream tasks")
        parser.add_argument("--down_l2", type=float, default=0, help="L2 penalty in downstream tasks")
    elif task == 'graph':
        parser.add_argument("--pooler", type=str, default="avg", help="Pooling function for graph embedding")
        parser.add_argument("--batch_size", type=int, default=32, help="The batch size of training")

    # Display settings
    args = parser.parse_args()

    return args


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        print("------------------ Best args not found, use default args ------------------")
    else:
        configs = configs[args.dataset]

        for k, v in configs.items():
            if "lr" in k or "beta" in k:
                v = float(v)
            setattr(args, k, v)
        print("------------------ Use best configs ------------------")

    return args


def print_config(config):
    print("------------------ Model Configuration ------------------")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("------------------ Model Configuration ------------------")


# ======================================================================
#   Logger functions
# ======================================================================


def create_logger(args, task):

    # Logger directory
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, args.dataset.lower()), exist_ok=True)
           
    model_info = '{}_{}_{}_{}'.format(args.dataset.lower(), args.kernel, args.num_layer, args.hid_dim)
    if args.single:
        model_info += '_single'
    else:
        model_info += '_{}'.format(args.dec_aggr)

    if args.nonskip:
        model_info += '_nonskip'
    if args.beta != 0:
        model_info += '_beta_{}'.format(str(args.beta))
    if args.norm:
        model_info += '_{}'.format(args.norm)
    
    if task == 'node':
        if args.emb_act:
            model_info += '_emb_act'
        
        log_file = 'log_' + model_info + '.txt'
    elif task == 'graph':
        log_file = 'log_' + model_info + '_pooler_{}.txt'.format(str(args.pooler))

    log_path = os.path.join(args.log_dir, args.dataset.lower(), log_file)
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    
    if args.log:
        log_handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        log_handlers = [
            logging.StreamHandler(sys.stdout)
        ]
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=log_handlers
    )
    logger = logging.getLogger()

    return logger, model_info


# ======================================================================
#   Graph normalization functions
# ======================================================================


def get_normalization(edge_index, num_nodes, improved=1., dtype=float, norm='sym', laplacian=True):
    """
    Obtain the Laplacian/Normalized matrix generated from normalized adjacency matrix
    (For PyTorch Geometric)
    """

    fill_value = improved

    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                            device=edge_index.device)

    # 'add_remaining_self_loop' moves all self-link to 
    # the end of list 
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    # assert tmp_edge_weight is not None
    # edge_weight = tmp_edge_weight

    # Normalize
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    
    if norm == 'sym':
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif norm == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

    if laplacian:
        # Laplacian
        tmp_edge_weight = edge_weight.new_full((edge_weight.size(0),), 0)
        tmp_edge_weight[-num_nodes:] = 1.
        edge_weight = tmp_edge_weight - edge_weight

    return edge_index, edge_weight


# ======================================================================
#   Model activation/normalization creation function
# ======================================================================

def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == 'relu':
        act = nn.ReLU()
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return norm


# ======================================================================
#   Model coefficient functions
# ======================================================================

def obtain_conv_coefs(kernel):
    """
    Return the coefficients of polynomial approximation of kernels
    """
    if kernel not in ['normal', 'heat', 'ppr']:
        raise ValueError("Invalid convolution kernel!")
    
    if kernel == 'normal':
        coefs = [1.0, -1.0]
    elif kernel == 'heat':
        coefs = [1.0, -1.0, 0.5, -1/6]
    
    return coefs


def conv_kernel_coefs(kernel, left=0., right=2.):
    """
    Construct encoder kernel function for approximation
    """
    if kernel == 'gcn':
        coefs = [1., -1.]
    elif kernel in ['gala', 'gcn_inv']:
        coefs = [1., 1.]
    elif kernel == 'gdn_enc':
        coefs = [1., -1., 0.5, -1/6]
    elif kernel == 'gdn_dec':
        coefs = [1., 1., 0.5, 1/6]
    elif kernel == 'heat':
        coefs = remez_enc_approx(approx_kernel_order(kernel), left, right, lambda x: np.exp(-x))
    elif kernel == 'ppr':
        coefs = remez_enc_approx(approx_kernel_order(kernel), left, right, lambda x: 1/(4*x + 1))
    elif kernel == 'heat_inv':
        coefs = remez_enc_approx(approx_kernel_order(kernel), left, right, lambda x: np.exp(x))
    elif kernel == 'ppr_inv':
        coefs = [1., 4.]
    
    return coefs


def approx_kernel_order(kernel):
    """
    Return the order of remez approximation of different kernel
    """
    if kernel == 'gcn':
        order = 9
    elif 'heat' in kernel:
        order = 2
    elif 'ppr' in kernel:
        order = 2
    
    return order


def wiener_kernel_func(x, kernel, penalty):
    """
    Construct wiener kernel function for approximation
    """
    if kernel not in ['gcn', 'heat', 'ppr']:
        raise ValueError("Invalid convolution kernel!")
    
    if kernel == 'gcn':
        conv = 1 - x
    elif kernel == 'heat':
        conv = np.exp(-x)
    elif kernel == 'ppr':
        conv = 1/(4*x + 1)
    
    return conv/(conv**2 + penalty)


def remez_init(n, left, right):
    """
    Chebyshev nodes for remez approximation
    """
    
    # Chebyshev nodes for initialization
    points = np.array([left] + [0]*n + [right])
    
    order = n+2
    for i in range(order):
        p = ((2*i + 1)/(2*order))*np.pi
        points[i] = (left+right)/2 + ((right-left)/2)*np.cos(p)

    points.sort()

    return points


def remez_step(n, points, func):
    """
    Coefficients calcuation
    """

    # Left-side of equation
    A = [np.repeat(1, n+2)]
    for i in range(n):
        A.append(A[i]*points)
    A.append([(-1)**(i%2) for i in range(n+2)])
    A = np.asarray(A).T

    # Right side of equation
    b = func(points)

    # Coefficients
    coefs = np.linalg.solve(A, b)

    return coefs[:-1]


def remez_enc_approx(n, left, right, func):
    """
    Compute the coefficients of nth-order remez polynomial approximation
    """

    # Initialization
    points = remez_init(n, left, right)

    # Compute coefficients
    coefs = remez_step(n, points, func)

    return coefs


def remez_approx(n, left, right, wiener_func, penalty):
    """
    Compute the coefficients of nth-order remez polynomial approximation
    """

    # Initialization
    points = remez_init(n, left, right)

    # Construct partial function
    func = partial(wiener_func, penalty=penalty)

    # Compute coefficients
    coefs = remez_step(n, points, func)

    return coefs


def scale_up(z, x_min, x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """
    
    return x_min + (z + 1) * (x_max - x_min) / 2


def cheb_approx(n, m, left, right, wiener_func, penalty):
    """
    Compute the coefficients of nth-order chebyshev polynomial approximation
    """

    # Initialization
    r_k = chebyshev.chebpts1(m)
    T = chebyshev.chebvander(r_k, n)

    # Construct partial function
    func = partial(wiener_func, penalty=penalty)

    # Fit
    points = scale_up(r_k, left, right)
    vals = func(points)
    coefs = np.linalg.inv(T.T @ T) @ T.T @ vals

    cheb_poly = chebyshev.Chebyshev(coefs, domain=[left, right])
    norm_poly = cheb_poly.convert(kind=Polynomial)
    coefs = norm_poly.coef

    return coefs