import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def clone_params(param, N):
    return nn.ParameterList([copy.deepcopy(param) for _ in range(N)])


# TODO: replaced with https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GraphLayer(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, num_of_nodes,
                 num_of_heads, dropout, alpha, concat=True):
        super(GraphLayer, self).__init__()
        self.in_features = in_features              # Embedding size
        self.hidden_features = hidden_features      # Embedding size
        self.out_features = out_features            # Embedding size
        self.alpha = alpha                          # For LeakyRELU ➔ hardcoded 0.1
        self.concat = concat                        # Encoder graph ➔ True; Decoder Graph ➔ False
        self.num_of_nodes = num_of_nodes            # Number of nodes in the graph ➔ patient features + 2
        self.num_of_heads = num_of_heads            # Number of attention heads ➔ 1 (VGNN/Mimic-III)
        
        # For VGNN/Mimic-III, 1 Attention head.
        self.W = clones(nn.Linear(in_features, hidden_features), num_of_heads)
        self.a = clone_params(nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True), num_of_heads)
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        
        if not concat:
            self.V = nn.Linear(hidden_features, out_features)
        else:
            self.V = nn.Linear(num_of_heads * hidden_features, out_features)
            
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        if concat:  # ???
            self.norm = LayerNorm(hidden_features)
        else:
            self.norm = LayerNorm(hidden_features)

    def initialize(self):
        for i in range(len(self.W)):
            nn.init.xavier_normal_(self.W[i].weight.data)
        for i in range(len(self.a)):
            nn.init.xavier_normal_(self.a[i].data)
        if not self.concat:
            nn.init.xavier_normal_(self.V.weight.data)
            nn.init.xavier_normal_(self.out_layer.weight.data)

    def attention(self, linear, a, N, data, edge):
        """Do convolution over the graph.

        Args:
            linear: weights (R^(dxd))
            a: bias (R^(1x(2*d)))
            N: number of nodes
            data: h_prime ➔ Embedding of Graph Nodes (N x Embedding_Size)
            edge: input_edges E.g.: (2 x 11664)
                            with 108 x 108 = 11664 ➔ 108 diag code/procedure code/lab value in a 'specific' patient

        Returns:
            h_prime ➔ Embedding of Graph Nodes (N x Embedding_Size)
        """
        data = linear(data).unsqueeze(0)
        assert not torch.isnan(data).any()
        # edge: 2*D x E
        h = torch.cat((data[:, edge[0, :], :], data[:, edge[1, :], :]), 
                      dim=0)
        data = data.squeeze(0)
        # h: N x out
        assert not torch.isnan(h).any()
        # edge_h: 2*D x E
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        # edge: 2*D x E
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.hidden_features * self.num_of_heads))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).to(device))
        # e_rowsum: N x 1
        row_check = (e_rowsum == 0) 
        e_rowsum[row_check] = 1
        zero_idx = row_check.nonzero()[:, 0]
        edge_e = edge_e.add(
            torch.sparse.FloatTensor(zero_idx.repeat(2, 1), torch.ones(len(zero_idx)).to(device), torch.Size([N, N])))  # type: ignore
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward(self, edge, data=None):
        """Forward pass for the GraphLayer.

        Args:
            edge: input_edges
            data: h_prime ➔ Embedding of Graph Nodes (N x Embedding_Size)

        Returns:
            _type_: _description_
        """
        N = self.num_of_nodes
        
        if self.concat: # hardcoded True
            # For VGNN/Mimic-III, 1 Attention head ➔ Zip has 1 element.
            h_prime = torch.cat([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=1)
        else:
            h_prime = torch.stack([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=0).mean(
                dim=0)
        
        h_prime = self.dropout(h_prime)
        
        if self.concat:
            return F.elu(self.norm(h_prime))
        else:
            return self.V(F.relu(self.norm(h_prime)))


class VariationalGNN(nn.Module):

    def __init__(self, 
                 in_features, 
                 out_features, 
                 num_of_nodes, 
                 n_heads, 
                 n_layers,
                 dropout, 
                 alpha,                     # For LeakyRELU ➔ hardcoded 0.1
                 variational=True, 
                 none_graph_features=0, 
                 concat=True):
        
        # Save input parameters for later convenient restoration of the object for inference.
        self.kwargs = {'in_features': in_features, 
                       'out_features': out_features, 
                       'num_of_nodes': num_of_nodes,
                       'n_heads': n_heads,
                       'n_layers': n_layers,
                       'dropout': dropout,
                       'alpha': alpha,
                       'variational': variational,
                       'none_graph_features': none_graph_features,
                       'concat': concat}
        
        super(VariationalGNN, self).__init__()
        self.variational = variational
        # Add 2 more nodes into the Graph: the 1st indicates the patient is of no conditions (diagnose code, proc code, lab values); the last node is used to absorb features from specific nodes of a specific patient, to make prediction for that patient.
        self.num_of_nodes = num_of_nodes + 2 - none_graph_features
        # The Graph! (N x embedding_features)
        self.embed = nn.Embedding(self.num_of_nodes, in_features, padding_idx=0)

        self.in_att = clones(
            GraphLayer(in_features, in_features, in_features, self.num_of_nodes,
                       n_heads, dropout, alpha, concat=True), n_layers)
        self.out_features = out_features
        self.out_att = GraphLayer(in_features, in_features, out_features, self.num_of_nodes,
                                  n_heads, dropout, alpha, concat=False)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.parameterize = nn.Linear(out_features, out_features * 2)
        self.out_layer = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1))
        self.none_graph_features = none_graph_features

        if none_graph_features > 0:
            self.features_ffn = nn.Sequential(
                nn.Linear(none_graph_features, out_features//2),
                nn.ReLU(),
                nn.Dropout(dropout))
            self.out_layer = nn.Sequential(
                nn.Linear(out_features + out_features//2, out_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_features, 1))

        for i in range(n_layers):
            self.in_att[i].initialize()


    def data_to_edges(self, data):
        """From the feature vector, output the edges for Encoder graph and Decode graph.
        Note: all Nodes are connected. But given a specific patient (with specific conditions), extract only related nodes corresponding to the conditions. Could be denoted as "mini-graph". The convolution will later take place only in this "mini-graph".

        Args:
            data: one feature vector for a patient ➔ (multi-hot encoding)

        Returns:
            _type_: (input_edges, output_edges)
                    input_edges: shape: (2, (number_of_conditions + 1) x (number_of_conditions + 1))
                    output_edges: shape: (2, (number_of_conditions + 2) x (number_of_conditions + 2))
        """
        length = data.size()[0]
        nonzero = data.nonzero()
        if nonzero.size()[0] == 0:  # Case when patient has zero conditions!
            # Right side: should include also torch.LongTensor([[0], [0]])???
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        if self.training:
            mask = torch.rand(nonzero.size()[0])
            mask = mask > 0.05
            nonzero = nonzero[mask]
            if nonzero.size()[0] == 0:  # ???
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        
        nonzero = nonzero.transpose(0, 1) + 1   # Why +1? ➔ Need to increase the original node indices by 1 (from patient feature vector) because now 2 more nodes are added (in the 1st position, and the last position).
        lengths = nonzero.size()[1]
        input_edges = torch.cat((nonzero.repeat(1, lengths),
                                 nonzero.repeat(lengths, 1).transpose(0, 1)
                                 .contiguous().view((1, lengths ** 2))), dim=0)

        nonzero = torch.cat((nonzero, torch.LongTensor([[length + 1]]).to(device)), dim=1)
        lengths = nonzero.size()[1]
        output_edges = torch.cat((nonzero.repeat(1, lengths),
                                  nonzero.repeat(lengths, 1).transpose(0, 1)
                                  .contiguous().view((1, lengths ** 2))), dim=0)
        return input_edges.to(device), output_edges.to(device)

    def reparameterise(self, mu, logvar):
        if self.training:
            # Assume: log_variation (NOT log_standard_deviation!)
            std = logvar.mul(0.5).exp_()
            # tensor.new() ➔ Constructs a new tensor of the same data type as self tensor.
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encoder_decoder(self, data):
        """Given a patient data, encode it into the total graph, then decode to the last node.

        Args:
            data ([N]): multi-hot encoding (of diagnose codes). E.g. shape = [1309]

        Returns:
            Tuple[Tensor, Tensor]: The last node's features, plus KL Divergence
        """
        N = self.num_of_nodes
        input_edges, output_edges = self.data_to_edges(data)
        h_prime = self.embed(torch.arange(N).long().to(device))
        
        # Encoder:
        for attn in self.in_att:    # Collection of GraphLayers
            h_prime = attn(input_edges, h_prime)
            
        if self.variational:
            # Even given only a patient's data, this parameterization affects the total graph. ➔ wasteful computation but doesn't affect the final outcome during back propagation.
            h_prime = self.parameterize(h_prime).view(-1, 2, self.out_features)
            h_prime = self.dropout(h_prime)
            mu = h_prime[:, 0, :]
            logvar = h_prime[:, 1, :]
            h_prime = self.reparameterise(mu, logvar)   # h_prime.shape = [N, z_dim] e.g. (1311x256)
            
            # Essential variables (mu, ,logvar) for computing DL Divergence later.
            # Note: only consider the patient's graph (NOT the total graph).
            split = int(math.sqrt(len(input_edges[0])))
            pat_diag_code_idx = input_edges[0][0:split]
            mu = mu[pat_diag_code_idx, :]
            logvar = logvar[pat_diag_code_idx, :]
            
        # Decoder:
        h_prime = self.out_att(output_edges, h_prime)
        
        if self.variational:
            """Need to divide with mu.size()[0] because the original formula sums over all latent dimensions.
            """
            return (h_prime[-1],            # The last node's features.
                    0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2)) / mu.size()[0]
                    )                       # KL Divergence.
        else:
            return (h_prime[-1], \
                    torch.tensor(0.0).to(device)
                    )

    def forward(self, data):
        # Concate batches
        batch_size = data.size()[0]
        # In eicu data the first feature whether have be admitted before is not included in the graph
        if self.none_graph_features == 0:   # hardcoded = 0! (Note: this does *not* mean "no node features")
            # For each Patient, encode the graph specifically for that.
            outputs = [self.encoder_decoder(data[i, :]) for i in range(batch_size)]
            # Return logits (output of self.out_layer()) ➔ later use BCEWithLogitsLoss (no need to convert to probability now).
            return self.out_layer(F.relu(torch.stack([out[0] for out in outputs]))), \
                   torch.sum(torch.stack([out[1] for out in outputs]))
        else:
            outputs = [(data[i, :self.none_graph_features],
                        self.encoder_decoder(data[i, self.none_graph_features:])) for i in range(batch_size)]
            return self.out_layer(F.relu(
                torch.stack([torch.cat((self.features_ffn(torch.FloatTensor([out[0]]).to(device)), out[1][0]))
                             for out in outputs]))), \
                   torch.sum(torch.stack([out[1][1] for out in outputs]), dim=-1)
