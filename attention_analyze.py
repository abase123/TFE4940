import torch

def analyze_window_attention(model, threshold, dataset, window_idx, device='cuda', node_num=5):
    x, _ = dataset[window_idx]
    x, _ = [item.float().to(device) for item in [x, _]]
    x = x.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        # Run the model and get predictions
        predictions = model(x, threshold=threshold, inference=True)

        # Extract attention weights
        attention_matrices = []
        for layer in model.gdn.gnn_layers:
            att_weight = layer.att_weight_1
            edge_index = layer.edge_index_1
            att_weight = torch.mean(att_weight, dim=1)
            
            # Reshape attention to node x node matrix
            att_matrix = torch.zeros(node_num, node_num)
            for i, (src, dst) in enumerate(edge_index.T):
                att_matrix[src % node_num, dst % node_num] = att_weight[i]
            
            attention_matrices.append(att_matrix)
        
        return predictions, attention_matrices, edge_index


def analyze_temporal_attention(model, dataset, start_idx, num_windows, threshold=0.2, device='cuda', node_num=36):
    temporal_attention = []
    predictions_list = []
    
    for i in range(num_windows):
        window_idx = start_idx + i
        if window_idx >= len(dataset):
            break

        # Get predictions, attention matrices, and edge indices for each window
        predictions, att_matrices, edge_index = analyze_window_attention(model, threshold, dataset, window_idx, device, node_num)
        predictions_list.append(predictions)
        temporal_attention.append(att_matrices)
        
    return predictions_list, temporal_attention, edge_index



