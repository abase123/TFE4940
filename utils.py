import plotly.graph_objects as go
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

import math

import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch 
from scipy.sparse import linalg
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""def train_and_eval(model, data_loader_train, data_loader_val, device=DEVICE, lr=0.001, use_pred=False, num_epoch=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch / num_epoch))

    train_loss_list = []
    val_loss_list = []
    last_epoch = False
    i = 0
    
    for epoch in range(num_epoch):
        if epoch == num_epoch - 1:
            last_epoch = True
        
        # Training Loop
        model.train()  # Ensure the model is in training mode
        total_train_loss = 0.0
        num_batches = 0

        for seq_x, seq_y in data_loader_train:
            seq_x, seq_y = [item.float().to(device) for item in [seq_x, seq_y]]
            print(seq_x.shape)
            optimizer.zero_grad()
            threshold = threshold_values[i] if i < len(threshold_values) else 0.2
            out = model(seq_x, threshold, last_epoch=last_epoch)
            loss = loss_func(out, seq_y) 
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
            

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / num_batches
        train_loss_list.append(avg_train_loss)
        print(f"Epoch: {epoch} - Average Training Loss: {avg_train_loss}, Threshold Value: {threshold}")
        
        # Step the learning rate scheduler
        scheduler.step()

        # Validation Loop
        avg_val_loss = eval(model, data_loader_val, device, threshold, last_epoch=last_epoch)
        val_loss_list.append(avg_val_loss)
        print(f"Epoch: {epoch} - Average Validation Loss: {avg_val_loss}")
        
        i += 1

    return model, train_loss_list, val_loss_list
    """

def get_fc_graph_struc(feature_list):
    struc_map = {}
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    
    return struc_map


def build_loc_net(struc, all_features, feature_map=[]):
    
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
        

    
    return edge_indexes

def adj_matrix_to_edge_index(adj_matrix):
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False).t()
    flipped_edge_indices = torch.stack((edge_indices[1], edge_indices[0]))  # Flip the rows
    return flipped_edge_indices

def edge_index_to_adj_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for i in range(edge_index.shape[1]):
        source, target = edge_index[:, i]
        adj_matrix[source, target] = 1
    return adj_matrix



def create_threshold_list(
    min_threshold_start = 0.1,# Starting minimum threshold value,
    max_threshold = 0.8,  # Maximum threshold value
    growth_rate = 0.05  ,     # Growth rate to control increase within the range
    min_increase_rate = 0.0005, # Rate at which the minimum threshold increases
    num_iterations = 200        # Total number of iterations to match the pattern
    ):
    # Generate threshold values with increasing minimum over time
    threshold_values = []
    current_min_threshold = min_threshold_start  # Initialize current minimum threshold

    for epoch in range(num_iterations):
        # Amplitude bounded by max_threshold and current minimum threshold
        amplitude = (max_threshold - current_min_threshold) * (1 - math.exp(-growth_rate * epoch))
        
        # Calculate cosine oscillation with increasing amplitude bounded by max_threshold
        threshold = current_min_threshold + amplitude * (0.5 * (1 + math.cos(math.pi * epoch / 5)))  # Adjust frequency
        threshold_values.append(threshold)
        
        # Incrementally increase the minimum threshold after each epoch
        current_min_threshold += min_increase_rate

    plt.figure(figsize=(12, 6))
    plt.plot(threshold_values, label=r"$\delta(t)$")
    plt.title(r"Dynamic Threshold $\delta(t)$ over Epochs", fontsize=24,fontweight="bold")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel(r"$\delta$", fontsize=18)
    plt.ylim(min_threshold_start, max_threshold)  # Keep within bounds
    plt.grid(alpha=0.7)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()

    return threshold_values


def get_mean_attn(temporal_att):
    mean_attn =  torch.zeros(len(list(df_mts.columns)), len(list(df_mts.columns)))
    for i in range(len(temporal_att)):
        mean_attn +=  temporal_att[i][0]

    mean_attn/=len(temporal_att)
    return mean_attn



def compute_transition(heatmap1, heatmap2):
    """
    Compute the transition matrix between two binary heatmaps.
    
    Args:
        heatmap1 (np.ndarray): The first binary heatmap (initial state).
        heatmap2 (np.ndarray): The second binary heatmap (next state).
        
    Returns:
        np.ndarray: A matrix showing the transitions.
                    1 for new connections, -1 for removed connections, 0 for unchanged.
    """
    # Ensure the heatmaps are binary (0 or 1) and have the same shape
    assert heatmap1.shape == heatmap2.shape, "Heatmaps must have the same shape"
    
    # Compute the transition matrix
    transition = heatmap2 - heatmap1
    
    return transition


def plot_pairs(df_mts,columns_to_plot_indices):
    # Define a list of colors to use for each time series
    colors = ['orange', 'blue']  # First plot will be red, second plot will be blue

    # Loop through each column index and create a separate figure for each
    for i, index in enumerate(columns_to_plot_indices):
        # Initialize a new figure for each time series
        fig = go.Figure()
        
        # Use color based on the order in the colors list
        color = colors[i % len(colors)]
        
        # Add the time series trace for the current column with the specified color
        fig.add_trace(go.Scatter(
            y=df_mts.iloc[:, index],
            mode='lines',
            name=f'Column {index}',
            line=dict(color=color)  # Set line color here
        ))
        
        # Add vertical lines to each plot
        fig.add_vline(x=300, line=dict(color='red', dash='dash', width=2))
        fig.add_vline(x=550, line=dict(color='green', dash='dash', width=2))

        # Set title and labels for each figure
        fig.update_layout(
            title=f'Time Series for Column {index}',
            xaxis_title='Time',
            yaxis_title='Value',
            legend_title="Series",
            font=dict(size=18),
            showlegend=True,
        )
        
        # Show each figure
        fig.show()

    # Create a combined plot with both time series in a single figure
    fig_combined = go.Figure()

    # Add both time series traces with different colors
    for i, index in enumerate(columns_to_plot_indices):
        color = colors[i % len(colors)]
        fig_combined.add_trace(go.Scatter(
            y=df_mts.iloc[:, index],
            mode='lines',
            name=f'Column {index}',
            line=dict(color=color)  # Set line color here
        ))

    # Add vertical lines to the combined plot
    fig_combined.add_vline(x=400, line=dict(color='red', dash='dash', width=2))
    #fig_combined.add_vline(x=400, line=dict(color='green', dash='dash', width=2))

    # Set title and labels for the combined figure
    fig_combined.update_layout(
        title='Combined Time Series for Selected Columns',
        xaxis_title='Time',
        yaxis_title='Value',
        legend_title="Series",
        font=dict(size=18),
        showlegend=True,
    )

    # Show the combined figure
    fig_combined.show()
    


def generate_neighbors_heatmap(model, time_series_data, top_n=10):
    """
    Generates a heatmap displaying each time series and its top neighbors.
    
    Args:
        model: The model instance that contains `get_top_neighbors`.
        time_series_data (pd.DataFrame): DataFrame where each column is a time series, and each row is a time step.
        top_n (int): Number of top neighbors to display for each time series.
    """
    # Number of time series
    num_series = time_series_data.shape[1]
    combined_data = []
    row_boundaries = []  # Track row boundaries for each time series and its neighbors

    for i in range(num_series):
        # Get the top N neighbors for the current time series (column)
        neighbors = model.get_top_neighbors(i, top_n=top_n,use_weighted=True)
        neighbor_indices = [i] + [neighbor[0] for neighbor in neighbors]  # Include the current time series first

        # Retrieve data for the current time series and its neighbors
        selected_series_data = time_series_data.iloc[:, neighbor_indices].T  # Transpose for correct format

        # Append each series (and its neighbors) as rows in combined_data
        combined_data.append(selected_series_data)

        # Update boundary position for current series and neighbors
        row_boundaries.append(len(combined_data))

    # Concatenate all series-neighbor blocks into one DataFrame for heatmap plotting
    heatmap_data = pd.concat(combined_data)

    # Plot heatmap
    plt.figure(figsize=(12, 25))
    sns.heatmap(heatmap_data, cmap='coolwarm', cbar=True,vmax=8)
    plt.xlabel("Time Steps")
    plt.ylabel("Time Series and Their Top Neighbors")
    plt.title("Top 10 Neighbors for Each Time Series")

    plt.show()



def plot_node_embeddings_tsne_plotly(model, node_num, device='cuda', width=1500, height=1000):
    # Safely retrieve and detach the node embeddings after training
    node_embeddings = model.embedding(torch.arange(node_num).to(device)).detach().cpu().numpy()
    
    # Optionally, apply the weight transformation without altering model parameters
    node_embeddings_weighted = node_embeddings * model.weight.detach().cpu().numpy()

    # Apply t-SNE to reduce dimensions to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(node_embeddings_weighted)

    # Create a DataFrame for the visualization
    df = pd.DataFrame({
        'Node': list(range(node_num)),
        'X': node_embeddings_2d[:, 0],
        'Y': node_embeddings_2d[:, 1]
    })

    # Create a Plotly scatter plot
    fig = px.scatter(
        df, x='X', y='Y', text='Node',
        title='t-SNE Visualization of Weighted Node Embeddings',
        labels={'X': 'Dimension 1', 'Y': 'Dimension 2'}
    )

    # Update layout for size and readability
    fig.update_traces(textposition='top center')
    fig.update_layout(hovermode='closest', width=width, height=height)

    # Show the figure
    fig.show()




def plot_node_embeddings_tsne_thesis(model, node_num, device='cuda', width=1900, height=1700):
    """
    Generate a t-SNE visualization of weighted node embeddings with jitter and enhanced styling for clarity.
    
    Parameters:
        model: The trained model containing node embeddings.
        node_num: Number of nodes to visualize.
        device: The device used for computations ('cuda' or 'cpu').
        width: Width of the output plot.
        height: Height of the output plot.
    """
    # Retrieve and process embeddings as before
    node_embeddings = model.embedding(torch.arange(node_num).to(device)).detach().cpu().numpy()
    node_embeddings_weighted = node_embeddings * model.weight.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    node_embeddings_2d = tsne.fit_transform(node_embeddings_weighted)

    # Create DataFrame
    df = pd.DataFrame({
        'Node': list(range(node_num)),
        'X': node_embeddings_2d[:, 0],
        'Y': node_embeddings_2d[:, 1]
    })

    # Add jitter to avoid overlapping
    jitter = 0.7  # Adjust jitter as needed
    df['X'] += np.random.uniform(-jitter, jitter, size=len(df))
    df['Y'] += np.random.uniform(-jitter, jitter, size=len(df))

    # Define highlighted nodes
    highlighted_nodes = {32, 6, 102}

    # Create Plotly scatter plot
    fig = go.Figure()

    # Plot non-highlighted nodes
    fig.add_trace(go.Scatter(
        x=df.loc[~df['Node'].isin(highlighted_nodes), 'X'],
        y=df.loc[~df['Node'].isin(highlighted_nodes), 'Y'],
        mode='markers+text',
        text=df.loc[~df['Node'].isin(highlighted_nodes), 'Node'],
        textposition='top center',
        marker=dict(
            size=14,  # Slightly larger size for better visibility
            color=df.loc[~df['Node'].isin(highlighted_nodes), 'Node'],
            colorscale='Viridis',  # Use a color scale
            showscale=True,
            colorbar=dict(
                title="Neuron",
                titlefont=dict(size=20, family='Arial', color='black'),
                tickfont=dict(size=14, family='Arial', color='black'),
                x=1.02  # Position colorbar
            )
        ),
        name='Nodes',
        showlegend=False  # Remove legend
    ))

    # Plot highlighted nodes
    fig.add_trace(go.Scatter(
        x=df.loc[df['Node'].isin(highlighted_nodes), 'X'],
        y=df.loc[df['Node'].isin(highlighted_nodes), 'Y'],
        mode='markers+text',
        text=df.loc[df['Node'].isin(highlighted_nodes), 'Node'],
        textposition='top center',
        marker=dict(
            size=15,  # Larger size for highlighted nodes
            color='red',  # Fixed color for highlights
            symbol='star'  # Star marker for highlighted nodes
        ),
        name='Highlighted Nodes',
        showlegend=False  # Remove legend
    ))

    # Update layout with styling
    fig.update_layout(
        title=dict(
            text='t-SNE Visualization of Neuron Embeddings',
            font=dict(size=40, family='Arial', color='black', weight='bold'),
            x=0.5
            
        ),
        xaxis=dict(
            title='Dimension 1',
            titlefont=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=14, family='Arial', color='black'),
            showgrid=True,
            zeroline=False,
            linecolor='black',
            linewidth=2.5
        ),
        yaxis=dict(
            title='Dimension 2',
            titlefont=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=14, family='Arial', color='black'),
            showgrid=True,
            zeroline=False,
            linecolor='black',
            linewidth=2.5
        ),
        hovermode='closest',
        width=width,
        height=height,
        template='plotly_white'  # Clean background for better visibility
    )

    fig.show()




def plot_node_embeddings_tsne_plotly_3d(model, node_num, device='cuda', width=1500, height=1000):
    # Safely retrieve and detach the node embeddings after training
    node_embeddings = model.embedding(torch.arange(node_num).to(device)).detach().cpu().numpy()
    
    # Optionally, apply the weight transformation without altering model parameters
    node_embeddings_weighted = node_embeddings * model.weight.detach().cpu().numpy()

    # Apply t-SNE to reduce dimensions to 3D for visualization
    tsne = TSNE(n_components=3, random_state=42)
    node_embeddings_3d = tsne.fit_transform(node_embeddings_weighted)

    # Create a DataFrame for the visualization
    df = pd.DataFrame({
        'Node': list(range(node_num)),
        'X': node_embeddings_3d[:, 0],
        'Y': node_embeddings_3d[:, 1],
        'Z': node_embeddings_3d[:, 2]
    })

    # Create a 3D Plotly scatter plot
    fig = px.scatter_3d(
        df, x='X', y='Y', z='Z', text='Node',
        title='3D t-SNE Visualization of Weighted Node Embeddings',
        labels={'X': 'Dimension 1', 'Y': 'Dimension 2', 'Z': 'Dimension 3'}
    )

    # Update layout for size and readability
    fig.update_traces(textposition='top center')
    fig.update_layout(hovermode='closest', width=width, height=height)

    # Show the figure
    fig.show()



import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch 
from scipy.sparse import linalg
from torch.autograd import Variable
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float().to(device)
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(device)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape).to(device)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(device)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y



def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reshape_edges(edges, N):
    #edges: B, N(N-1), type
    edges = torch.tensor(edges)
    mat = torch.zeros(edges.shape[-1], edges.shape[0], N, N) +1
    mask = ~torch.eye(N, dtype = bool).unsqueeze(0).unsqueeze(0)
    mask = mask.repeat(edges.shape[-1], edges.shape[0], 1, 1)
    mat[mask] = edges.permute(2, 0, 1).flatten()
    return mat

def load_data_train(filename, DEVICE, batch_size, shuffle=True):
    file_data = np.load(filename)
    train_x = file_data['train_x']
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']


    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())


    return train_loader, train_target_tensor, val_loader, val_target_tensor, mean, std



def load_data_test(filename, DEVICE, batch_size, shuffle=False):
    file_data = np.load(filename)
    train_x = file_data['test_x']
    train_target = file_data['test_target']
    train_x = train_x[:, :, 0:1, :]



    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)


    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


    return train_loader, train_target_tensor,  mean, std



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def point_adjust_eval(anomaly_start, anomaly_end, down, loss, thr1, thr2):
    len_ = int(anomaly_end[-1]/down)+1
    anomaly = np.zeros((len_))
    loss = loss[:len_]
    anomaly[np.where(loss>thr1)] = 1
    anomaly[np.where(loss<thr2)] = 1
    ground_truth = np.zeros((len_))
    for i in range(len(anomaly_start)):
        ground_truth[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)+1] = 1
        if np.sum(anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)])>0:
            anomaly[int(anomaly_start[i]/down) :int(anomaly_end[i]/down)] = 1
        anomaly[int(anomaly_start[i]/down) ] = ground_truth[int(anomaly_start[i]/down) ]
        anomaly[int(anomaly_end[i]/down)] = ground_truth[int(anomaly_end[i]/down)]

    return anomaly, ground_truth



