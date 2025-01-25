import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F



from graph_layer import GraphLayer


def get_batch_edge_index(edge_index, batch_size, nodes_per_batch):
    """
    Create batched edge indices by shifting the node indices for each batch.
    
    :param edge_index: Tensor (2, num_edges) containing the edge indices.
    :param batch_size: Number of batches.
    :param nodes_per_batch: Number of nodes in each batch.
    :return: Tensor (2, batch_size * num_edges) containing the batched edge indices.
    """
    # Clone and repeat edge indices for each batch
    edge_index = edge_index.clone().detach()
    num_edges = edge_index.shape[1]
    batched_edge_index = edge_index.repeat(1, batch_size).contiguous()

    # Offset node indices for each batch
    for batch_id in range(batch_size):
        batched_edge_index[:, batch_id * num_edges : (batch_id + 1) * num_edges] += batch_id * nodes_per_batch

    return batched_edge_index.long()



class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, out_window, inter_num = 512):
        super(OutLayer, self).__init__()

        self.out_window = out_window
        self.node_num = node_num
        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, out_window))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        print(out.shape)

        out = out.view(out.size(0), self.node_num, self.out_window)

        return out

        
class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=0):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=True)
        self.ff = nn.Linear(out_channel * heads, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.attn_weigths = []
        self.edge_indecies = []

    def forward(self, x, edge_index, embedding=None, node_num=0):
        
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
       
        self.att_weight_1 = att_weight
        print(att_weight.shape)
        self.attn_weigths.append(att_weight)
        self.edge_index_1 = new_edge_index
        self.edge_indecies.append(new_edge_index)
        print("in gnnLayer")
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.ff(out) 
        out = self.bn(out)
        print(out.shape)
        return self.relu(out)



class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, use_pred = False, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, out_win=200, topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.weight_vector = nn.Parameter(torch.Tensor(embed_dim))


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=4) for i in range(1)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        
        self.use_pred = use_pred

        self.out_layer_1 = OutLayer(dim*edge_set_num, node_num, out_layer_num, out_window = out_win, inter_num = out_layer_inter_dim)
        

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

        self.all_topk_indices = []
        self.all_cos_scores = []

        #self.adjacency_matrices = []


    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def cosine_similarity(self,embeddings):
        """
        Compute the cosine similarity matrix for a set of node embeddings.
        
        :param embeddings: Tensor (num_nodes, embedding_dim)
        :return: Tensor (num_nodes, num_nodes) containing cosine similarity between each pair of nodes.
        """
        # Compute the dot product between node embeddings
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Normalize the dot product using the L2 norms of the embeddings
        norms = embeddings.norm(dim=-1, keepdim=True)
        normed_matrix = torch.matmul(norms, norms.T)

        return similarity_matrix / normed_matrix


    def create_topk_edges(similarity_matrix, topk):
        """
        Create a top-k adjacency matrix based on the cosine similarity matrix.
        
        :param similarity_matrix: Tensor (num_nodes, num_nodes) containing cosine similarity values.
        :param topk: Number of top-k neighbors to select for each node.
        :return: Edge indices (2, num_edges) representing the top-k neighbors.
        """
        topk_indices = torch.topk(similarity_matrix, topk, dim=-1)[1]

        source_nodes = torch.arange(similarity_matrix.size(0)).unsqueeze(1).repeat(1, topk).flatten()
        destination_nodes = topk_indices.flatten()
        
        edge_index = torch.stack([destination_nodes, source_nodes], dim=0)
        
        return edge_index
        
    def forward(self, data, last_epoch=False):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        print("in model:")
        print("x")
        print(x.shape)
        gcn_outs = []

        edge_index = self.edge_index_sets[0]
        edge_num = edge_index.shape[1]
        cache_edge_index = self.cache_edge_index_sets[0]
        
        if cache_edge_index is None or cache_edge_index.shape[0] != edge_num*batch_num:
            self.cache_edge_index_sets[0] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
        
        batch_edge_index = self.cache_edge_index_sets[0]
        
        all_embeddings = self.embedding(torch.arange(node_num).to(device))

        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)

        weights = weights_arr.view(node_num, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / normed_mat
        
        dim = weights.shape[-1]
        topk_num = self.topk

        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        if last_epoch:
            # Save the top_k indices and cosine similarity matrix for visualization later
            self.all_topk_indices.append(topk_indices_ji.cpu().detach().numpy())
            self.all_cos_scores.append(cos_ji_mat.cpu().detach().numpy())

        self.learned_graph = topk_indices_ji
    
        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        print(gated_i.shape)
        print(gated_j.shape)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        print("edge_batch:")
        print(gated_edge_index.shape)
        batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)        
        
        gcn_out = self.gnn_layers[0](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

        print("ferdig")
        gcn_outs.append(gcn_out)
        
        print(len(gcn_outs))
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        print(x.shape)

        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        print("out:")
        print(out.shape)

        out = self.dp(out)
        out_1 = self.out_layer_1(out)
        
        return out_1



"""
class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, use_pred = False, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, out_win=200, topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        #device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=4) for i in range(1)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        
        self.use_pred = use_pred

        self.out_layer_1 = OutLayer(dim*edge_set_num, node_num, out_layer_num, out_window = input_dim, inter_num = out_layer_inter_dim)
        
        if (use_pred):
            self.out_layer_2 = OutLayer(dim*edge_set_num, node_num, out_layer_num, out_window = out_win, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

        self.all_topk_indices = []
        self.all_cos_scores = []

        #self.adjacency_matrices = []

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        
    def forward(self, data, last_epoch=False):
        #batch_num, node_num, all_feature = data.shape
        #x = data.view(batch_num, node_num, -1)
        #x = self.conv_layer(x) 

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        print("in model:")
        print("x")
        print(x.shape)
        gcn_outs = []

        edge_index = self.edge_index_sets[0]
        edge_num = edge_index.shape[1]
        cache_edge_index = self.cache_edge_index_sets[0]
        
        if cache_edge_index is None or cache_edge_index.shape[0] != edge_num*batch_num:
            self.cache_edge_index_sets[0] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
        
        batch_edge_index = self.cache_edge_index_sets[0]
        
        all_embeddings = self.embedding(torch.arange(node_num).to(device))

        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)

        weights = weights_arr.view(node_num, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / normed_mat
        
        dim = weights.shape[-1]
        topk_num = self.topk

        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        if last_epoch:
            # Save the top_k indices and cosine similarity matrix for visualization later
            self.all_topk_indices.append(topk_indices_ji.cpu().detach().numpy())
            self.all_cos_scores.append(cos_ji_mat.cpu().detach().numpy())

        self.learned_graph = topk_indices_ji
    
        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        print(gated_i.shape)
        print(gated_j.shape)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        print("edge_batch:")
        print(gated_edge_index.shape)
        batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)        
        
        gcn_out = self.gnn_layers[0](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
        #gcn_out = self.gnn_layers[0](x, batch_gated_edge_index, node_num=node_num, embedding=all_embeddings)

        print("ferdig")
        gcn_outs.append(gcn_out)
        
        print(len(gcn_outs))
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        print(x.shape)

        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        print("out:")
        print(out.shape)

        out = self.dp(out)
        out_1 = self.out_layer_1(out)
        #out = out.view(-1, node_num)
        
        if(self.use_pred):
            out_2 = self.out_layer_2(out)
            return out_1, out_2

        return out_1

"""
    
"""    def forward(self, data, save_adjency = False):
        # Embed all nodes
        batch_num, node_num, feature_dim = data.shape
        x = data.clone().detach()
        device = data.device

        # Flatten data for GNN input
       
        node_indices = torch.arange(node_num).to(device)
        node_embeddings = self.embedding(node_indices)  # Shape: (node_num, embed_dim)

        
        # Link prediction
        link_probs = torch.zeros(node_num, node_num).to(device)
        for i in range(node_num):
            for j in range(node_num):
                if i != j:
                    # Concatenate the time series features of nodes i and j
                    combined_features = torch.cat([x[:, i, :], x[:, j, :]], dim=-1)  # (batch_num, feature_dim * 2)
                    # Average across batch to get one value for the entire batch (you can also use another strategy here)
                    combined_features_mean = combined_features.mean(dim=0)  # (feature_dim * 2)
                    link_probs[i, j] = self.link_prediction_layer(combined_features_mean)
    

        # Threshold link probabilities to generate adjacency matrix
        adj_matrix = (link_probs > 0.6).float()  # Threshold at 0.5 to get discrete edges

        # Ensure symmetry (if undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0.6).float()
        
        if(save_adjency):
            self.adjacency_matrices.append(adj_matrix.cpu().detach().numpy())

        # Generate edge index from adjacency matrix
        edge_indices = adj_matrix.nonzero(as_tuple=False).t()  # Shape: (2, num_edges)
        batch_edge_index = get_batch_edge_index(edge_indices, batch_num, node_num).to(device)

        # Forward pass through GNN layer
        all_embeddings = node_embeddings.repeat(batch_num, 1)
        x = data.view(-1, feature_dim).contiguous()

        gcn_out = self.gnn_layers[0](x, batch_edge_index, embedding=all_embeddings, node_num=node_num * batch_num)
        x = gcn_out.view(batch_num, node_num, -1)

        # Final prediction
        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(out)
        out = out.permute(0, 2, 1)

        out_1 = self.out_layer_1(out)

        return out_1
        
    """







