import sys
import logging
from sklearn import decomposition
from sklearn import datasets
import os
from os.path import join as pjoin
import gae.utility as utility
import matplotlib.pylab as plt
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
        self.alpha = 0.1
        self.learning_info1 = {}
        self.learning_info2 = {}
        self.learning_info3 = {}
        self.learning_info4 = {}
        self.learning_info5 = {}
        self.dict = {}
        self.vectors={}

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.nets, self.dict = utility.read_nets(
            self.net_input, self.weighted, self.directed, self.log)
        self.hierarchy = utility.read_hierarchy(self.hierarchy_input, self.log)

    def relabel_nodes(self):
        new_nets = {}
        for net_name, net in self.nets.items():
            def mapping(x):
                return '%s__%d' % (net_name, x)
            new_nets[net_name] = nx.relabel_nodes(net, mapping, copy=False)
        return new_nets

    def update_internal_vectors(
            self, all_nodes, leaf_vectors, internal_vectors):
        new_internal_vectors = {}
        for hnode in self.hierarchy.nodes():
            if hnode in self.nets:
                # leaf vectors are optimized separately
                continue
            self.log.info('Updating internal hierarchy node: %s' % hnode)
            new_internal_vectors[hnode] = {}
            children = list(self.hierarchy.successors(hnode))
            parents = list(self.hierarchy.predecessors(hnode))
            for node in all_nodes:
                # update internal vectors (see Eq. 4 in Gopal et al.)
                if parents:
                    t1 = 1. / (len(children) + 1.)
                else:
                    t1 = 1. / len(children)
                t2 = [np.zeros(leaf_vectors.values()[0].shape)]
                for child in children:
                    if child in self.nets:
                        node_name = '%s__%s' % (child, node)
                        # node can be missing in certain networks
                        if node_name in leaf_vectors:
                            t2.append(leaf_vectors[node_name])
                    else:
                        t2.append(internal_vectors[child][node])
                if parents:
                    parent = parents[0]
                    assert len(parents) == 1, 'Problem'
                    parent_vector = internal_vectors[parent][node]
                else:
                    # root of the hierarchy
                    parent_vector = 0
                new_internal_vector = t1 * (parent_vector + sum(t2))
                new_internal_vectors[hnode][node] = new_internal_vector
        return new_internal_vectors

    def get_all_nodes(self):
        all_nodes = set()
        for _, net in self.nets.items():
            # nodes = [node.split('__')[1] for node in net.nodes()]
            nodes = [node for node in net.nodes()]
            all_nodes.update(nodes)
        self.log.info('All nodes: %d' % len(all_nodes))
        return list(all_nodes)

    def init_internal_vectors(self, all_nodes):
        internal_vectors = {}
        for hnode in self.hierarchy.nodes():
            if hnode in self.nets:
                # leaf vectors are optimized separately
                continue
            internal_vectors[hnode] = {}
            for node in all_nodes:
                vector = (self.rng.rand(self.dimension) - 0.5) / self.dimension
                internal_vectors[hnode][node] = vector
            n_vectors = len(internal_vectors[hnode])
            self.log.info('Hierarchy node: %s -- %d' % (hnode, n_vectors))
        return internal_vectors

    def get_leaf_vectors(self, model):
        leaf_vectors = {}
        for word, val in model.vocab.items():
            leaf_vector = model.syn0[val.index]
            assert type(word) == str, 'Problems with vocabulary'
            leaf_vectors[word] = leaf_vector
        return leaf_vectors

    def gcn_multilayer(self):
        """Neural embedding of a multilayer network"""
        # self.nets = self.relabel_nodes()
        all_nodes = self.get_all_nodes()
        # internal_vectors = self.init_internal_vectors(all_nodes)
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
            # temp = self.get_leaf_vectors(model)

            self.vectors[net_name] = sess.run(model.embeddings, feed_dict=feed_dict)
            # internal_vectors = self.update_internal_vectors(all_nodes, leaf_vectors, internal_vectors)
            # fname = self.out_dir + net_name +'_vectors.txt'
            # np.savetxt(fname, np.array(leaf_vectors), fmt="%s", delimiter='  ')
            #
            # self.log.info('Saving vectors: %s' % fname)
            # ==============================================================
            self.log.info('after exec gcn : %s' % net_name)

        self.log.info('Done!')
        np.savetxt(self.out_dir + 'ALL_Nodes.txt', all_nodes, fmt="%s", delimiter='  ')

        # fname = pjoin(self.out_dir, 'internal_vectors.emb')
        # self.log.info('Saving internal vectors: %s' % fname)

        # return self.model

    def gcn_plot(self):
        lists1 = sorted(self.learning_info1.items())  # sorted by key, return a list of tuples
        lists2 = sorted(self.learning_info2.items())  # sorted by key, return a list of tuples
        lists3 = sorted(self.learning_info3.items())  # sorted by key, return a list of tuples
        lists4 = sorted(self.learning_info4.items())  # sorted by key, return a list of tuples
        lists5 = sorted(self.learning_info5.items())  # sorted by key, return a list of tuples

        x1, y1 = zip(*lists1)  # unpack a list of pairs into two tuples
        x2, y2 = zip(*lists2)  # unpack a list of pairs into two tuples
        x3, y3 = zip(*lists3)  # unpack a list of pairs into two tuples
        x4, y4 = zip(*lists4)  # unpack a list of pairs into two tuples
        # x5, y5 = zip(*lists5)  # unpack a list of pairs into two tuples

        plt.subplot(2, 2, 1)
        plt.plot(x1, y1, 'tab:orange')
        plt.xlabel('Iteration')
        plt.ylabel('Train Loss')

        plt.subplot(2, 2, 2)
        plt.plot(x2, y2, 'tab:green')
        plt.xlabel('Iteration')
        plt.ylabel('Average Accuracy')

        plt.subplot(2, 2, 3)
        plt.plot(x3, y3, 'tab:blue')
        plt.xlabel('Iteration')
        plt.ylabel('ROC Score')

        plt.subplot(2, 2, 4)
        plt.plot(x4, y4, 'tab:red')
        plt.xlabel('Iteration')
        plt.ylabel('Average Precision')

        # plt.subplot(2, 3, 5)
        # plt.plot(x5, y5, 'tab:orange')
        # plt.xlabel('iteration')
        # plt.ylabel('Undamped')

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # fig.suptitle('Sharing x per column, y per row')
        # ax1.plot(x, y)
        # ax2.plot(x, y ** 2, 'tab:orange')
        # ax3.plot(x, -y, 'tab:green')
        # ax4.plot(x, -y ** 2, 'tab:red')
        #
        # for ax in fig.get_axes():
        #     ax.label_outer()

        plt.show()

    def gcn_predict(self):
        output = os.path.join('gae', 'emb')
        output2 = os.path.join('gae', 'data')
        RootVectors = {}
        for net_name, net in self.nets.items():
            for item in self.vectors[net_name]:
                RootVectors.update(item)
        pca = decomposition.PCA(n_components=16)
        output = pca.fit_transform(RootVectors)
        output.shape

        mappings = self.read_labels('collapsedNetworkdict.txt', output, 0)
        labels1 = self.read_labels('brain_GO_0007420.txt', os.path.join(output2, 'brain-label'), 1)
        labels2 = self.read_labels('brain_GO_0021885.txt', os.path.join(output2, 'brain-label'), 1)
        labels3 = self.read_labels('brain_GO_0022029.txt', os.path.join(output2, 'brain-label'), 1)
        labels4 = self.read_labels('brain_GO_0030900.txt', os.path.join(output2, 'brain-label'), 1)
        labels5 = self.read_labels('brain_GO_0030901.txt', os.path.join(output2, 'brain-label'), 1)

        from sklearn import model_selection as mf
        from sklearn import svm
        svm_model = svm.SVC(kernel='sigmoid')

        Y1 = [0] * RootVectors.shape[0]
        Y2 = [0] * RootVectors.shape[0]
        Y3 = [0] * RootVectors.shape[0]
        Y4 = [0] * RootVectors.shape[0]
        Y5 = [0] * RootVectors.shape[0]
        for key, index in sorted(mappings.items()):
            Y1[index] = labels1[key]
            Y2[index] = labels2[key]
            Y3[index] = labels3[key]
            Y4[index] = labels4[key]
            Y5[index] = labels5[key]

        X = RootVectors
        results1 = mf.cross_validate(svm_model, X, Y1, cv=10,
                                     scoring=['precision_macro', 'recall_macro', 'roc_auc'])
        results2 = mf.cross_validate(svm_model, X, Y2, cv=10,
                                     scoring=['precision_macro', 'recall_macro', 'roc_auc'])
        results3 = mf.cross_validate(svm_model, X, Y3, cv=10,
                                     scoring=['precision_macro', 'recall_macro', 'roc_auc'])
        results4 = mf.cross_validate(svm_model, X, Y4, cv=10,
                                     scoring=['precision_macro', 'recall_macro', 'roc_auc'])
        results5 = mf.cross_validate(svm_model, X, Y5, cv=10,
                                     scoring=['precision_macro', 'recall_macro', 'roc_auc'])
        # results['test_recall_macro']
        # results['test_roc_auc']
        print(np.mean(results1['test_roc_auc']), np.mean(results1['test_precision_macro']),
              np.mean(results1['test_recall_macro']))
        print(np.mean(results2['test_roc_auc']), np.mean(results2['test_precision_macro']),
              np.mean(results2['test_recall_macro']))
        print(np.mean(results3['test_roc_auc']), np.mean(results3['test_precision_macro']),
              np.mean(results3['test_recall_macro']))
        print(np.mean(results4['test_roc_auc']), np.mean(results4['test_precision_macro']),
              np.mean(results4['test_recall_macro']))
        print(np.mean(results5['test_roc_auc']), np.mean(results5['test_precision_macro']),
              np.mean(results5['test_recall_macro']))


