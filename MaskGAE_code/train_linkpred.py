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

def load_or_initialize_statistics(data):
    """Load existing statistics or initialize for the first time."""
    if os.path.exists(STATS_FILE):
        print(f"Loading existing statistics from {STATS_FILE}")
        return pd.read_csv(STATS_FILE)
    else:
        # Initialize statistics with each link in the dataset
        print(f"Initializing new statistics for all links.")
        edge_index = data.edge_index.cpu().numpy().T  # All edges in the dataset
        df = pd.DataFrame(edge_index, columns=['source', 'target'])
        df['appeared_in_test'] = 0  # Number of times it appeared in the test set
        df['correctly_predicted'] = 0  # Number of times it was predicted correctly
        df['total_predictions'] = 0  # Total number of predictions made for the link
        return df

def save_statistics(statistics_df):
    """Save the updated statistics to a CSV file."""
    statistics_df.to_csv(STATS_FILE, index=False)
    print(f"Statistics saved to {STATS_FILE}")

def update_statistics(statistics_df, link_info):
    # Print the edges in statistics_df where the source or target is one of the nodes 0-9
    print("Edges from statistics_df where source or target is 0-9:")
    mask = (statistics_df['source'].isin(range(10))) | (statistics_df['target'].isin(range(10)))
    print(statistics_df[['source', 'target']][mask])
    
    # Print the edges in link_info where the source or target is 0-9
    print("\nEdges from link_info where source or target is 0-9:")
    for i, link in enumerate(link_info):
        if link['source'] in range(10) or link['target'] in range(10):
            print(f"Link {i}: source={link['source']}, target={link['target']}")

def train_linkpred(model, splits, args, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    
    # Load or initialize the statistics for links in the dataset
    statistics_df = load_or_initialize_statistics(test_data)

    test_link_info = None  # Initialize before the epoch loop

    for epoch in tqdm(range(1, 1 + args.epochs)):

        loss = model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
        
        if epoch % args.eval_period == 0:
            valid_auc, valid_ap, valid_correct_pred, valid_restored_links, valid_link_info = model.test_step(valid_data, 
                                                  valid_data.pos_edge_label_index, 
                                                  valid_data.neg_edge_label_index, 
                                                  batch_size=batch_size)
            if valid_auc > best_valid:
                best_valid = valid_auc
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(torch.load(args.save_path))
    test_results = model.test_step(test_data, 
                                    test_data.pos_edge_label_index, 
                                    test_data.neg_edge_label_index, 
                                    batch_size=batch_size)
    test_auc, test_ap, test_correct_pred, test_restored_links, test_link_info = test_results

    # Check if test_link_info was updated
    if test_link_info is not None:
        # After all epochs in a single run, update the statistics once
        update_statistics(statistics_df, test_link_info)
        # print(statistics_df.head(100))
        print(f"Run {run} completed and statistics updated.")
    else:
        print(f"No test data processed for run {run}.")

    # Save the updated statistics at the end of the iteration
    # save_statistics(statistics_df)

    return test_auc, test_ap, test_correct_pred, test_restored_links



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="Cora", help="Datasets. (default: Cora)")
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

data = get_dataset(root, args.dataset, transform=transform)

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
