import argparse
import numpy as np
import pandas as pd
import os  # For checking if statistics file exists
import torch
import torch_geometric.transforms as T
from tqdm.auto import tqdm

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from maskgae.mask import MaskEdge, MaskPath

# Path to the statistics file
STATS_FILE = "link_statistics.csv"

def load_or_initialize_statistics(train_data, valid_data, test_data):
    """
    Load existing statistics from a file or initialize a new DataFrame to track statistics for all links.
    
    Args:
        train_data (Data): Training data containing edge indices.
        valid_data (Data): Validation data containing edge indices.
        test_data (Data): Testing data containing positive and negative edge indices.
    
    Returns:
        pandas.DataFrame: A DataFrame containing source and target nodes, as well as statistical counters.
    """
    if os.path.exists(STATS_FILE):
        print(f"Loading existing statistics from {STATS_FILE}")
        return pd.read_csv(STATS_FILE)
    else:
        # Initialize statistics with all links from train, valid, test positive, and test negative edges
        print(f"Initializing new statistics for all links.")
        
        # Combine edges from train, validation, and test sets (positive and negative)
        edge_index_train_valid = torch.cat([train_data.edge_index, valid_data.edge_index], dim=1)
        edge_index_test_pos = test_data.pos_edge_label_index
        edge_index_test_neg = test_data.neg_edge_label_index
        
        # Combine all the edges into a single tensor
        edge_index = torch.cat([edge_index_train_valid, edge_index_test_pos, edge_index_test_neg], dim=1)
        edge_index = edge_index.cpu().numpy().T  # Convert to NumPy and transpose
        
        # Create DataFrame
        df = pd.DataFrame(edge_index, columns=['source', 'target'])
        df['appeared_in_test'] = 0  # Number of times it appeared in the test set
        df['correctly_predicted'] = 0  # Number of times it was predicted correctly
        df['total_predictions'] = 0  # Total number of predictions made for the link
        return df


def save_statistics(statistics_df):
    """
    Save the updated statistics DataFrame to a CSV file.

    Args:
        statistics_df (pandas.DataFrame): The DataFrame containing updated link statistics.
    """
    statistics_df.to_csv(STATS_FILE, index=False)
    print(f"Statistics saved to {STATS_FILE}")

def update_statistics(statistics_df, link_info):
    """
    Update the statistics for each link based on the current run's test results.

    Args:
        statistics_df (pandas.DataFrame): The existing statistics DataFrame.
        link_info (list of dict): A list containing link information with keys:
            - 'source': Source node ID
            - 'target': Target node ID
            - 'appeared': Number of times the link appeared in the test set (usually 1).
            - 'correct': Whether the link was predicted correctly (1 for correct, 0 otherwise).
    """
    # Convert source/target in statistics_df to int64 for consistency
    statistics_df['source'] = statistics_df['source'].astype(np.int64)
    statistics_df['target'] = statistics_df['target'].astype(np.int64)

    # Iterate over all links in the current test set
    for link in link_info:
        # Explicitly convert the source and target in link_info to int64
        source = np.int64(link['source'])
        target = np.int64(link['target'])
        appeared, correct = link['appeared'], link['correct']

        # Handle undirected edges by checking both directions
        mask = ((statistics_df['source'] == source) & (statistics_df['target'] == target)) | \
               ((statistics_df['source'] == target) & (statistics_df['target'] == source))

        # Update counters (for the matching link) that corresponds rows in the DataFrame
        statistics_df.loc[mask, 'appeared_in_test'] += appeared
        statistics_df.loc[mask, 'correctly_predicted'] += correct
        statistics_df.loc[mask, 'total_predictions'] += 1

def train_linkpred(model, splits, args, device="cpu"):
    """
    Train and evaluate the link prediction model.

    Args:
        model (Model): The MaskGAE model to train and evaluate.
        splits (dict): A dictionary containing train, valid, and test data splits.
        args (Namespace): Command-line arguments for training configuration.
        device (str): The device to use ('cpu' or 'cuda').
    
    Returns:
        Tuple: Test AUC, AP, correct predictions, and restored links.
    """
    # Initialize optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    # Load data splits onto the specified device
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    # Load or initialize the statistics for links in the dataset
    statistics_df = load_or_initialize_statistics(train_data, valid_data, test_data)

    test_link_info = None  # Initialize before the epoch loop

    for epoch in tqdm(range(1, 1 + args.epochs)):
        # Perform one training step
        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        # Evaluate on validation data periodically
        if epoch % args.eval_period == 0:
            valid_auc, valid_ap, valid_correct_pred, valid_restored_links, valid_link_info = model.test_step(valid_data, 
                                                  valid_data.pos_edge_label_index, 
                                                  valid_data.neg_edge_label_index, 
                                                  batch_size=batch_size)
            # Save the model if validation performance improves
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(args.save_path))
    test_results = model.test_step(test_data, 
                                    test_data.pos_edge_label_index, 
                                    test_data.neg_edge_label_index, 
                                    batch_size=batch_size)
    test_auc, test_ap, test_correct_pred, test_restored_links, test_link_info = test_results

    # Update statistics based on the test run
    if test_link_info is not None:
        # After all epochs in a single run, update the statistics once
        update_statistics(statistics_df, test_link_info)
        print(f"Run {run} completed and statistics updated.")
    else:
        print(f"No test data processed for run {run}.")

    # Save the updated statistics at the end of the iteration
    save_statistics(statistics_df)

    return test_auc, test_ap, test_correct_pred, test_restored_links



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora) 'Custom' for custom dataset")
parser.add_argument("--custom_dataset_path", nargs="?", default="", help="Path to the custom dataset CSV file (required for Custom dataset).")
parser.add_argument("--mask", nargs="?", default="Path", help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu", help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2**16, help='Number of batch size. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="edge", help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument("--save_path", nargs="?", default="MaskGAE-LinkPred.pt", help="save path for model. (default: MaskGAE-LinkPred.pt)")
parser.add_argument("--device", type=int, default=0)


try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])


# (!IMPORTANT) Specify the path to your dataset directory ##############
# root = '~/public_data/pyg_data' # my root directory
root = 'data/'
########################################################################

data = get_dataset(root, args.dataset, transform=transform, custom_path=args.custom_dataset_path)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.3,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)

splits = dict(train=train_data, valid=val_data, test=test_data)

if args.mask == 'Path':
    mask = MaskPath(p=args.p, num_nodes=data.num_nodes, 
                    start=args.start,
                    walk_length=args.encoder_layers+1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None # vanilla GAE

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

if args.decoder_layers == 0:
    edge_decoder = DotEdgeDecoder()
else:
    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)


model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

auc_results = []
ap_results = []
correct_pred_results = []
restored_links_results = []

for run in range(1, args.runs+1):
    test_auc, test_ap, test_correct_pred, test_restored_links = train_linkpred(model, splits, args, device=device)
    auc_results.append(test_auc)
    ap_results.append(test_ap)
    correct_pred_results.append(test_correct_pred)
    restored_links_results.append(test_restored_links)
    print(f'Run {run} - AUC: {test_auc:.2%}, AP: {test_ap:.2%}, Correct Predictions: {test_correct_pred}, Restored Links: {test_restored_links}')   

print(f'Link Prediction Results ({args.runs} runs):\n'
      f'AUC: {np.mean(auc_results):.2%} ± {np.std(auc_results):.2%}',
      f'AP: {np.mean(ap_results):.2%} ± {np.std(ap_results):.2%}',
      f'Correct Predictions: {np.mean(correct_pred_results)} ± {np.std(correct_pred_results)}',
      f'Restored Links: {np.mean(restored_links_results)} ± {np.std(restored_links_results)}',
     )
