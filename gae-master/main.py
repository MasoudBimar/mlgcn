import argparse
import tensorflow as tf
from gae import mlgcn

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Run MlGCN')

    parser.add_argument('--input', nargs='?', default='gae/testdata/brain.list',
                        help='Path to a file containing locations of network layers')

    parser.add_argument('--outdir', nargs='?', default='gae/emb/',
                        help='Path to a directory where results are saved')

    parser.add_argument('--hierarchy', nargs='?', default='gae/testdata/brain.hierarchy',
                        help='Path to a file containing multi-layer network hierarchy')

    parser.add_argument('--dimension', type=int, default=18,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--iter', default=FLAGS.epochs, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.add_argument('--learning_rate', type=float, default=FLAGS.learning_rate, help='Initial learning rate.')
    parser.add_argument('--hidden1', type=int, default=FLAGS.hidden1, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=FLAGS.hidden2, help='Number of units in hidden layer 2.')
    parser.add_argument('--weight_decay', type=float, default=FLAGS.weight_decay, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--dropout', type=float, default=FLAGS.dropout, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--model', type=str, default='gcn_ae', help='Model string.')
    # parser.add_argument('dataset', 'cora', 'Dataset string.')
    parser.add_argument('--features',type=int,default=FLAGS.features, help='Whether to use features (1) or not (0).')
    parser.set_defaults(directed=False)
    return parser.parse_args()


def main(args):
    on = mlgcn.MlGCN(
        net_input=args.input, weighted=args.weighted, directed=args.directed,
        hierarchy_input=args.hierarchy,
        dimension=args.dimension,
        n_iter=args.iter,
        out_dir=args.outdir, model_str=args.model, features=args.features,
        dropout=args.dropout, weight_decay=args.weight_decay,
        hidden1=args.hidden1, hidden2=args.hidden2, learning_rate=args.learning_rate)
    on.gcn_multilayer()


args = parse_args()
main(args)

