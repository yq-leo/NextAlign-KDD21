from scipy.special import expit
from sklearn.metrics.pairwise import manhattan_distances
import torch
import numpy as np


def test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, test_set, anchor_links2, dist, mode='training'):
    '''
    Testing phase.
    @param model: current model
    @param topk: metrics - hits@k
    @param g: input merged graph
    @param x: output node embedding matrix
    @param edge_types: edge types of merged graph
    @param node_mapping1: original node indices of G1 to node indices in merged graph
    @param node_mapping2: original node indices of G2 to node indices in merged graph
    @param test_set: testing data
    @param anchor_links2: anchor nodes in G2
    @param dist: distance type
    @param mode: indicate training or testing
    @return:
        hits: hit@k scores
    '''
    model.eval()
    node_mapping1 = node_mapping1.cpu().detach().numpy()
    node_mapping2 = node_mapping2.cpu().detach().numpy()

    test_nodes1, test_nodes2 = test_set[:, 0], test_set[:, 1]
    with torch.no_grad():
        out_x = model(g, x, edge_types).cpu().detach().numpy()
        dim = out_x.shape[1]
        weights = model.score_lin.weight[0].cpu().detach().numpy()
        x1 = out_x[test_set[:, 0]]
        x2 = out_x[node_mapping2]

        x1_1 = x1[:, :dim//2]
        x1_2 = x1[:, dim//2: dim]
        x2_1 = x2[:, :dim//2]
        x2_2 = x2[:, dim//2: dim]
        if dist == 'inner':
            S = weights[0] * x1_1.dot(x2_1.T) + weights[1] * x1_1.dot(x2_2.T) + weights[2] * x1_2.dot(x2_1.T) + \
                weights[3] * x1_2.dot(x2_2.T)
        else:
            S = weights[0] * manhattan_distances(x1_1, x2_1) + \
                weights[1] * manhattan_distances(x1_1, x2_2) + \
                weights[2] * manhattan_distances(x1_2, x2_1) + \
                weights[3] * manhattan_distances(x1_2, x2_2)
            S = -S
        S = expit(S)
        if mode == 'testing':
            for i in range(len(S)):
                S[i][anchor_links2] = 0
        idx2 = np.argsort(-S, axis=1)[:, :topk[-1]]
        idx2_complete = np.argsort(-S, axis=1)

        test_set = set(tuple(i) for i in test_set)
        hits_l = []
        for k in topk:
            id2 = idx2[:, :k].reshape((-1, 1))
            idx1 = np.repeat(test_nodes1.reshape((-1, 1)), k, axis=1).reshape(-1, 1)
            idx = np.concatenate([idx1, id2], axis=1)
            idx = set(tuple(i) for i in idx)
            count = len(idx.intersection(test_set))
            hit = round(count/len(test_set), 4)
            hits_l.append(hit)
        # MRR
        mrr_l = 0
        anchor_l2r = {v1: v2 for (v1, v2) in test_set}
        for i, v1 in enumerate(test_nodes1):
            mrr_l += 1 / (1 + np.argwhere(idx2_complete[i, :] == anchor_l2r[v1]).flatten()[0])
        mrr_l = round(mrr_l / len(test_set), 4)

        # G2 to G1
        x1 = out_x[node_mapping1]
        x2 = out_x[node_mapping2[test_nodes2]]
        x1_1 = x1[:, :dim // 2]
        x1_2 = x1[:, dim // 2: dim]
        x2_1 = x2[:, :dim // 2]
        x2_2 = x2[:, dim // 2: dim]
        if dist == 'inner':
            S = weights[0] * x1_1.dot(x2_1.T) + weights[1] * x1_1.dot(x2_2.T) + weights[2] * x1_2.dot(x2_1.T) + \
                weights[3] * x1_2.dot(x2_2.T)
        else:
            S = weights[0] * manhattan_distances(x1_1, x2_1) + \
                weights[1] * manhattan_distances(x1_1, x2_2) + \
                weights[2] * manhattan_distances(x1_2, x2_1) + \
                weights[3] * manhattan_distances(x1_2, x2_2)
            S = -S
        S = expit(S)
        idx2 = np.argsort(-S, axis=0)[:topk[-1], :]
        idx2_complete = np.argsort(-S, axis=0)

        test_set = set(tuple(i) for i in test_set)
        hits_r = []
        for k in topk:
            id2 = idx2[:k, :].reshape((-1, 1))
            idx1 = np.repeat(test_nodes2.reshape((1, -1)), k, axis=0).reshape((-1, 1))
            idx = np.concatenate([id2, idx1], axis=1)
            idx = set(tuple(i) for i in idx)
            count = len(idx.intersection(test_set))
            hit = round(count / len(test_set), 4)
            hits_r.append(hit)
        # MRR
        mrr_r = 0
        anchor_r2l = {v2: v1 for (v1, v2) in test_set}
        for j, v2 in enumerate(test_nodes2):
            mrr_r += 1 / (1 + np.argwhere(idx2_complete[:, j] == anchor_r2l[v2]).flatten()[0])
        mrr_r = round(mrr_r / len(test_set), 4)

        hits_l = np.array(hits_l)
        hits_r = np.array(hits_r)
        hits = np.maximum(hits_l, hits_r)
        mrr = max(mrr_l, mrr_r)

    return hits, mrr


def rm_out(arr):
    """
    Remove outliers from an array
    :param arr: input array
    :return:  without outliers
    """
    arr = np.sort(arr)
    if len(arr) <= 3:
        return arr
    num_rm = len(arr) // 2
    return arr[int(num_rm / 2) + num_rm % 2: -(num_rm // 2)]
