# Wiener Graph Deconvolutional Network Improves Self-Supervised Learning

This is the official implementation of the following paper:

[Wiener Graph Deconvolutional Network Improves Self-Supervised Learning](xxxxxxx)  
*Jiashun Cheng, Man Li, Jia Li, Fugee Tsung; AAAI 2023*

Dependencies
----------------------
- python >= 3.7
- torch >= 1.11.0
- torch_geometric >= 2.0.3
- ogb >= 1.3.4
- argparse >= 1.1.0
- numpy >= 1.12.2
- scikit_learn >= 1.0.2
- scipy >= 1.4.1

Quick Start
----------------------

To reproduce the reported results, please run the script with `--use_cfg`.

**Node classification**

```
# With best configurations
python main_node.py --dataset PubMed --use_cfg

# Or you can customize the configurations (such as propagation kernel, decoder aggregation and etc.)
python main_node.py --dataset PubMed --kernel heat --dec_aggr sum 
```

Supported datasets includes `Cora`, `CiteSeer`, `PubMed`, `CS`, `Physics`, `Computers`, `Photo`, `ogbn-arxiv`

**Graph classification**

```
# With best configurations
python main_graph.py --dataset IMDB_BINARY --use_cfg --seed 2 12 22 32 42

# Or you can customize the configurations (such as propagation kernel, decoder aggregation, pooler and etc.)
python main_node.py --dataset IMDB_BINARY --kernel heat --dec_aggr sum --pooler max --seed 2 12 22 32 42
```

Supported datasets includes `IMDB-BINARY`, `IMDB-MULTI`, `PROTEINS`, `COLLAB`, `DD`, `NCI1`

Citing
----------------------

If you find this work is helpful to your research, please consider citing our paper:

```
@article{cheng2022wgdn,
  title={Wiener Graph Deconvolutional Network Improves Self-Supervised Learning},
  author={Cheng, Jiashun and Li, Man and Li, Jia and Tsung, Fugee},
  journal={xx},
  pages={xx},
  year={2022}
}
```
