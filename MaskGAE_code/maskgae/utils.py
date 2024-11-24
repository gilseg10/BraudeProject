import sys
import os
import torch
import random
import numpy as np
import pandas as pd
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_dataset(root: str, name: str, transform=None, custom_path: str = None) -> Data:
    """
    Get a graph dataset, supporting built-in and custom datasets.

    Args:
        root (str): Root directory for datasets.
        name (str): Name of the dataset (e.g., 'Cora', 'Custom').
        transform (callable, optional): Transformation to apply to the dataset.
        custom_path (str, optional): Path to the custom dataset CSV file (used only for 'Custom').

    Returns:
        Data: PyTorch Geometric Data object.
    """
    if name in {'arxiv', 'products', 'mag'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{name}')
        if name in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                    x=rel_data.x_dict['paper'],
                    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                    y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train']['paper'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid']['paper'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test']['paper'], data.num_nodes)
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
            data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
            data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    elif name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, name)
        data = transform(dataset[0])

    elif name == 'Reddit':
        dataset = Reddit(osp.join(root, name))
        data = transform(dataset[0])
    elif name in {'Photo', 'Computers'}:
        dataset = Amazon(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name in {'CS', 'Physics'}:
        dataset = Coauthor(root, name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif name == 'Custom': 
        if not custom_path:
            raise ValueError("Custom dataset selected but no 'custom_path' provided.")
        csv_path = os.path.join(root, custom_path)  # Use the custom dataset path
        print(f"Loading custom dataset from {csv_path}...")
        data = load_custom_dataset(csv_path)  # Load the custom dataset
        if transform:
            data = transform(data)  # Apply transformations if provided
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
    return data

def load_custom_dataset(csv_path: str) -> Data:
    """
    Load a custom dataset from a CSV file and preprocess it into a PyTorch Geometric Data object.
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
    Returns:
        Data: PyTorch Geometric Data object containing the graph structure, node features, and labels.
    """
    # Load the dataset from the CSV file
    df = pd.read_csv(csv_path)

    # Extract edges (source and target nodes)
    edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)

    # Extract node features (assuming feature columns are feature_1 and feature_2)
    node_features = df[['feature_1', 'feature_2']].values
    x = torch.tensor(node_features, dtype=torch.float)

    # Extract labels
    labels = df['label'].values
    y = torch.tensor(labels, dtype=torch.long)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data

def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()
