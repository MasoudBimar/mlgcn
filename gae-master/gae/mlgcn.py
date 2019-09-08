import sys
import logging
import os
from os.path import join as pjoin
import gae.utility as utility

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time
import os
# # Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

__version__ = '0.1'


class MlGCN():
    def __init__(self, net_input, weighted, directed,
                 hierarchy_input,
                 dimension, n_iter, out_dir,
                 model_str, features, dropout, weight_decay,
                 hidden1, hidden2, learning_rate, seed=0):
        self.net_input = net_input
        self.weighted = weighted
        self.directed = directed
        self.hierarchy_input = hierarchy_input
        self.dimension = dimension
        self.n_iter = n_iter
        self.out_dir = out_dir
        self.rng = np.random.RandomState(seed)
        self.log = logging.getLogger('MlGCN')
        self.model_str = model_str
        self.features = features
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.learning_rate = learning_rate

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.nets = utility.read_nets(
            self.net_input, self.weighted, self.directed, self.log)
        self.hierarchy = utility.read_hierarchy(self.hierarchy_input, self.log)

    def relabel_nodes(self):
        new_nets = {}
        for net_name, net in self.nets.items():
            def mapping(x):
                return '%s__%d' % (net_name, x)
            new_nets[net_name] = nx.relabel_nodes(net, mapping, copy=False)
        return new_nets

    def get_all_nodes(self):
        all_nodes = set()
        for _, net in self.nets.items():
            # nodes = [node.split('__')[1] for node in net.nodes()]
            nodes = [node for node in net.nodes()]
            all_nodes.update(nodes)
        self.log.info('All nodes: %d' % len(all_nodes))
        return list(all_nodes)

    def get_leaf_vectors(self, model):
        leaf_vectors = {}
        for word, val in model.vocab.items():
            leaf_vector = model.syn0[val.index]
            assert type(word) == str, 'Problems with vocabulary'
            leaf_vectors[word] = leaf_vector
        return leaf_vectors

    def gcn_multilayer(self):
        """Neural embedding of a multilayer network"""
        self.nets = self.relabel_nodes()
        all_nodes = self.get_all_nodes()
        tmp_fname = pjoin(self.out_dir, 'tmp.emb')
        for net_name, net in self.nets.items():
            self.log.info('Run GCN For Net: %s' % net_name)
            # =============================================================
            adjacency_matrix = nx.adjacency_matrix(net)
            adjacency_matrix = adjacency_matrix.todense()
            nodes_count = adjacency_matrix.shape[0]
            adj = adjacency_matrix
            features = sp.identity(nodes_count)
            adj = sp.csr_matrix(adj)
            # ----------------myCode-----------------------------------
            # Store original adjacency matrix (without diagonal entries) for later
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()
            # tst_actual_matrix = adj.toarray()
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
            adj = adj_train
            # -----------------------------myCode-------------------------
            # if FLAGS.features == 0:
            #    features = sp.identity(features.shape[0])  # featureless
            # -----------------------------myCode-------------------------
            # Some pre processing
            adj_norm = preprocess_graph(adj)
            # Define placeholders
            placeholders = {
                'features': tf.sparse_placeholder(tf.float32),
                'adj': tf.sparse_placeholder(tf.float32),
                'adj_orig': tf.sparse_placeholder(tf.float32),
                'dropout': tf.placeholder_with_default(0., shape=())
            }
            num_nodes = adj.shape[0]
            features = sparse_to_tuple(features.tocoo())
            num_features = features[2][1]
            features_nonzero = features[1].shape[0]
            # Create model
            model = None
            if self.model_str == 'gcn_ae':
                model = GCNModelAE(placeholders, num_features, features_nonzero, self.hidden1, self.hidden2)
            elif self.model_str == 'gcn_vae':
                model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, self.hidden1, self.hidden2)

            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            # Optimizer
            with tf.name_scope('optimizer'):
                if self.model_str == 'gcn_ae':
                    opt = OptimizerAE(preds=model.reconstructions,
                                      labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                  validate_indices=False), [-1]),
                                      pos_weight=pos_weight,
                                      norm=norm)
                elif self.model_str == 'gcn_vae':
                    opt = OptimizerVAE(preds=model.reconstructions,
                                       labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                   validate_indices=False), [-1]),
                                       model=model, num_nodes=num_nodes,
                                       pos_weight=pos_weight,
                                       norm=norm)

            # Initialize session
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            cost_val = []
            acc_val = []

            def get_roc_score(edges_pos, edges_neg, emb=None):
                if emb is None:
                    feed_dict.update({placeholders['dropout']: 0})
                    emb = sess.run(model.z_mean, feed_dict=feed_dict)

                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                # Predict on test set of edges
                adj_rec = np.dot(emb, emb.T)
                preds = []
                pos = []
                for e in edges_pos:
                    preds.append(sigmoid(adj_rec[e[0], e[1]]))
                    pos.append(adj_orig[e[0], e[1]])

                preds_neg = []
                neg = []
                for e in edges_neg:
                    preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                    neg.append(adj_orig[e[0], e[1]])

                preds_all = np.hstack([preds, preds_neg])
                labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
                roc_score = roc_auc_score(labels_all, preds_all)
                ap_score = average_precision_score(labels_all, preds_all)

                return roc_score, ap_score

            cost_val = []
            acc_val = []
            val_roc_score = []
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label = sparse_to_tuple(adj_label)
            # Train model
            # for epoch in range(FLAGS.epochs):
            # epochs = 10
            dropout = 0
            for epoch in range(self.n_iter):
                self.log.info('Iteration: %d' % epoch)
                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
                # feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # -----------myCode------------
                feed_dict.update({placeholders['dropout']: dropout})
                # -----------myCode------------
                # Run single weight update
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

                # Compute average loss
                avg_cost = outs[1]
                avg_accuracy = outs[2]

                roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
                val_roc_score.append(roc_curr)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                      "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                      "val_ap=", "{:.5f}".format(ap_curr),
                      "time=", "{:.5f}".format(time.time() - t))

            print("Optimization Finished!")
            roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
            print('Test ROC score: ' + str(roc_score))
            print('Test AP score: ' + str(ap_score))

            # ------vector generation -----------------------------
            vectors = sess.run(model.embeddings, feed_dict=feed_dict)
            fname = self.out_dir + net_name +'_vectors.txt'
            np.savetxt(fname, np.array(vectors), fmt="%s", delimiter='  ')

            self.log.info('Saving vectors: %s' % fname)
            # ==============================================================
            self.log.info('after exec gcn : %s' % net_name)

        self.log.info('Done!')
        np.savetxt(self.out_dir + 'ALL_Nodes.txt', all_nodes, fmt="%s", delimiter='  ')

        # fname = pjoin(self.out_dir, 'internal_vectors.emb')
        # self.log.info('Saving internal vectors: %s' % fname)

        # return self.model
