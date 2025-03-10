import torch
import functools, math
import torch.nn as nn
# from dgl import function as fn
# from dgl.nn.pytorch import utils
# from dgl.base import DGLError

"""
class RelGCN_dgl(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=True, activation=None, self_loop=True, dropout=0.0, alpha=0.5, param=True):
        '''
        RelGCN layer for network alignment.

        @param in_feat: input feature dimension.
        @param out_feat: output feature dimension.
        @param num_rels: number of relations. For two input graphs, num_rels=2.
        @param bias: whether to apply bias. Default is True.
        @param activation: whether to apply activation function. Default is None.
        @param self_loop: whether to apply self loops. Default is True.
        @param dropout: dropout rate. Default is 0.
        @param alpha: hyper-parameter in Eq. (6).
        @param param: whether to apply weight matrices.
        '''

        super(RelGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout
        self.alpha = alpha
        self.param = param

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        self.message_func = self.base_message_func

        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def base_message_func(self, edges, etypes):
        '''
        Message passing function.

        @param edges: edges in the input graph.
        @param etypes: edge types in the graph, indicating which graph the edges belong to.
        @return:
            msg: messages that will be passed along edges.
        '''
        weight = self.weight
        h = edges.src['h']

        if h.dtype == torch.int64 and h.ndim == 1:
            weight = weight.view(-1, weight.shape[2])
            flat_idx = etypes * weight.shape[1] + h
            msg = weight.index_select(0, flat_idx)
        else:
            if self.param:
                weight = weight.index_select(0, etypes)
                msg = torch.bmm(h.unsqueeze(1), weight).squeeze(1)
            else:
                msg = h

        return {'msg': msg}

    def forward(self, g, feat, etypes):
        '''
        Forward pass of RelGCN layer.

        @param g: input merged graph.
        @param feat: input features.
        @param etypes: edge types of merged graph.
        @return:
            node_repr: node embedding matrix.
        '''
        if isinstance(etypes, torch.Tensor):
            if len(etypes) != g.num_edges():
                raise DGLError('"etypes" tensor must have length equal to the number of edges'
                               ' in the graph. But got {} and {}.'.format(
                    len(etypes), g.num_edges()))

        with g.local_scope():
            g.srcdata['h'] = feat
            if self.self_loop:
                if self.param:
                    loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                                                             self.loop_weight)
                else:
                    loop_message = feat[:g.number_of_dst_nodes()]
            # message passing
            g.update_all(functools.partial(self.message_func, etypes=etypes),
                         fn.sum(msg='msg', out='h'))
            node_repr = g.dstdata['h'] * math.sqrt(self.alpha)

            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message * math.sqrt(1 - self.alpha)
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)

            return node_repr

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree


class RelGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=True, activation=None, self_loop=True, dropout=0.0, alpha=0.5, param=True):
        super(RelGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout
        self.alpha = alpha
        self.param = param

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, x, edge_type):
        """
        Forward pass of the RelGCN layer without DGL.

        @param x: Node features (num_nodes, in_feat).
        @param edge_index: Edge list (2, num_edges).
        @param edge_type: Edge types (num_edges).
        @return:
            node_repr: Node embedding matrix (num_nodes, out_feat).
        """

        # Sanity check for edge types
        if isinstance(edge_type, torch.Tensor):
            if edge_type.size(0) != edge_index.size(1):
                raise ValueError(f'"edge_type" tensor must have length equal to the number of edges. '
                                 f'Got {edge_type.size(0)} and {edge_index.size(1)}.')

        num_nodes = x.size(0)

        # Self-loop handling
        if self.self_loop:
            if self.param:
                if x.dtype == torch.int64:
                    loop_message = self.loop_weight.index_select(0, x[:num_nodes])
                else:
                    loop_message = torch.matmul(x[:num_nodes], self.loop_weight)
            else:
                loop_message = x[:num_nodes]
        else:
            loop_message = torch.zeros_like(x)

        # Message Passing
        src, dst = edge_index
        msg = self.message(x[src], edge_type)  # Call the message function for each edge

        # Aggregation (similar to DGL's fn.sum)
        # aggregated_msg = scatter_add(msg, dst, dim=0, dim_size=num_nodes)
        aggregated_msg = torch.zeros((num_nodes, *msg.shape[1:]), dtype=msg.dtype, device=msg.device)
        aggregated_msg.index_add_(0, dst, msg)

        # Feature Fusion (scaling with sqrt(alpha))
        node_repr = aggregated_msg * math.sqrt(self.alpha)

        # Adding bias if present
        if self.bias:
            node_repr += self.h_bias

        # Adding self-loop contribution
        if self.self_loop:
            node_repr += loop_message * math.sqrt(1 - self.alpha)

        # Applying activation function if specified
        if self.activation:
            node_repr = self.activation(node_repr)

        # Applying dropout
        node_repr = self.dropout(node_repr)

        return node_repr

    def message(self, x_j, edge_type):
        """
        Message passing function without DGL.

        @param x_j: Source node features (num_edges, in_feat).
        @param edge_type: Edge types (num_edges), indicating which relation the edge belongs to.
        @return:
            msg: Messages to be passed along edges (num_edges, out_feat).
        """
        weight = self.weight  # Shape: (num_rels, in_feat, out_feat)

        # Case 1: When input features are integer IDs (e.g., for embedding lookups)
        if x_j.ndim == 1:
            weight = weight.view(-1, weight.shape[2])  # Reshape to (num_rels * in_feat, out_feat)
            flat_idx = edge_type * weight.shape[1] + x_j.to(torch.int64)  # Compute flattened index
            msg = weight.index_select(0, flat_idx)  # Select corresponding weights
        else:
            # Case 2: When input features are real-valued (e.g., continuous node features)
            if self.param:
                selected_weight = weight[edge_type]  # Select relation-specific weights
                msg = torch.bmm(x_j.unsqueeze(1), selected_weight).squeeze(1)  # Batch matrix multiplication
            else:
                msg = x_j  # If no transformation, pass features as-is

        return msg

