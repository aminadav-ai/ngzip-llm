import torch
import torch.nn as nn
import numpy as np

class LlamaCompiledMLP(nn.Module):
    def __init__(self, original_mlp, layer_idx, recommendations, dummy_data, device, model_dtype, max_depth, max_mse, tree_cache=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.model_dtype = model_dtype
        self.max_depth = max_depth
        self.tree_cache = tree_cache if tree_cache is not None else {}
        
        self.tree_indices = []
        self.orig_indices = []
        
        for j in range(8192):
            method = recommendations.get((layer_idx, j), 'Original')
            if method == 'Decision Tree':
                self.tree_indices.append(j)
            else:
                self.orig_indices.append(j)
                
        self.W_gate_orig = None
        self.W_up_orig = None
        self.W_down_orig = None
        
        if self.orig_indices:
            self.orig_idx_t = torch.tensor(self.orig_indices, dtype=torch.long, device=device)
            if isinstance(original_mlp, LlamaCompiledMLP) or isinstance(original_mlp, LlamaDualTreeCompiledMLP):
                old_positions = [original_mlp.orig_indices.index(j) for j in self.orig_indices]
                self.W_gate_orig = nn.Parameter(original_mlp.W_gate_orig.data[old_positions].clone(), requires_grad=True)
                self.W_up_orig = nn.Parameter(original_mlp.W_up_orig.data[old_positions].clone(), requires_grad=True)
                self.W_down_orig = nn.Parameter(original_mlp.W_down_orig.data[:, old_positions].clone(), requires_grad=True)
            else:
                self.W_gate_orig = nn.Parameter(original_mlp.gate_proj.weight.data[self.orig_indices].clone(), requires_grad=True)
                self.W_up_orig = nn.Parameter(original_mlp.up_proj.weight.data[self.orig_indices].clone(), requires_grad=True)
                self.W_down_orig = nn.Parameter(original_mlp.down_proj.weight.data[:, self.orig_indices].clone(), requires_grad=True)
            
        self.tree_features = None
        self.tree_thresholds = None
        self.tree_leaves = None
        self.W_down_tree = None
        
        if self.tree_indices:
            self.tree_idx_t = torch.tensor(self.tree_indices, dtype=torch.long, device=device)
            
            tree_features_list = []
            tree_thresholds_list = []
            tree_leaves_list = []
            
            for j in self.tree_indices:
                feats, thres, lvs = self.tree_cache[j]
                tree_features_list.append(feats)
                tree_thresholds_list.append(thres)
                tree_leaves_list.append(lvs)
                
            self.tree_features = nn.Parameter(torch.tensor(tree_features_list, dtype=torch.long, device=device), requires_grad=False)
            self.tree_thresholds = nn.Parameter(torch.tensor(tree_thresholds_list, dtype=torch.float32, device=device), requires_grad=False)
            self.tree_leaves = nn.Parameter(torch.tensor(tree_leaves_list, dtype=torch.float32, device=device), requires_grad=False)
            
            if isinstance(original_mlp, LlamaCompiledMLP) or isinstance(original_mlp, LlamaDualTreeCompiledMLP):
                W_down_tree_data = torch.zeros(2048, len(self.tree_indices), dtype=model_dtype, device=device)
                for col_idx, tree_neuron_idx in enumerate(self.tree_indices):
                    if tree_neuron_idx in original_mlp.tree_indices:
                        old_col_idx = original_mlp.tree_indices.index(tree_neuron_idx)
                        W_down_tree_data[:, col_idx] = original_mlp.W_down_tree.data[:, old_col_idx]
                    else:
                        old_col_idx = original_mlp.orig_indices.index(tree_neuron_idx)
                        W_down_tree_data[:, col_idx] = original_mlp.W_down_orig.data[:, old_col_idx]
                self.W_down_tree = nn.Parameter(W_down_tree_data, requires_grad=True)
            else:
                self.W_down_tree = nn.Parameter(original_mlp.down_proj.weight.data[:, self.tree_indices].clone(), requires_grad=True)
            
        self.act_fn = nn.SiLU()
        self.replaced_count = len(self.tree_indices)
        
    def forward(self, x):
        sh = x.shape
        x_flat = x.view(-1, 2048)
        N_samples = x_flat.size(0)
        
        out_flat = torch.zeros(N_samples, 2048, dtype=x.dtype, device=x.device)
        
        if self.orig_indices:
            z_gate = torch.matmul(x_flat, self.W_gate_orig.T)
            z_up = torch.matmul(x_flat, self.W_up_orig.T)
            a_orig = self.act_fn(z_gate) * z_up
            out_flat += torch.matmul(a_orig, self.W_down_orig.T)
            
        if self.tree_indices:
            N_trees = len(self.tree_indices)
            num_splits = (2 ** self.max_depth) - 1
            
            x_features = x_flat[:, self.tree_features.view(-1)].view(N_samples, N_trees, num_splits)
            decisions = (x_features > self.tree_thresholds)
            
            curr_node = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
            for _ in range(self.max_depth):
                dec = torch.gather(decisions, 2, curr_node.unsqueeze(2)).squeeze(2)
                curr_node = torch.where(dec, 2 * curr_node + 2, 2 * curr_node + 1)
                
            leaf_offset = num_splits
            leaf_idx = curr_node - leaf_offset
            
            a_tree = torch.gather(self.tree_leaves, 1, leaf_idx.T).T
            a_tree = a_tree.to(x.dtype)
            out_flat += torch.matmul(a_tree, self.W_down_tree.T)
            
        return out_flat.view(sh[0], sh[1], 2048)


class LlamaDualTreeCompiledMLP(nn.Module):
    def __init__(self, original_mlp, layer_idx, recommendations, dummy_data, device, model_dtype, max_depth, max_mse, tree_cache=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        self.model_dtype = model_dtype
        self.max_depth = max_depth # Normally 4 for each tree, making it 16 leaves each
        self.tree_cache = tree_cache if tree_cache is not None else {}
        
        self.tree_indices = []
        self.orig_indices = []
        
        for j in range(8192):
            method = recommendations.get((layer_idx, j), 'Original')
            if method == 'Decision Tree':
                self.tree_indices.append(j)
            else:
                self.orig_indices.append(j)
                
        self.W_gate_orig = None
        self.W_up_orig = None
        self.W_down_orig = None
        
        if self.orig_indices:
            self.orig_idx_t = torch.tensor(self.orig_indices, dtype=torch.long, device=device)
            if isinstance(original_mlp, LlamaCompiledMLP) or isinstance(original_mlp, LlamaDualTreeCompiledMLP):
                old_positions = [original_mlp.orig_indices.index(j) for j in self.orig_indices]
                self.W_gate_orig = nn.Parameter(original_mlp.W_gate_orig.data[old_positions].clone(), requires_grad=True)
                self.W_up_orig = nn.Parameter(original_mlp.W_up_orig.data[old_positions].clone(), requires_grad=True)
                self.W_down_orig = nn.Parameter(original_mlp.W_down_orig.data[:, old_positions].clone(), requires_grad=True)
            else:
                self.W_gate_orig = nn.Parameter(original_mlp.gate_proj.weight.data[self.orig_indices].clone(), requires_grad=True)
                self.W_up_orig = nn.Parameter(original_mlp.up_proj.weight.data[self.orig_indices].clone(), requires_grad=True)
                self.W_down_orig = nn.Parameter(original_mlp.down_proj.weight.data[:, self.orig_indices].clone(), requires_grad=True)
            
        self.gate_tree_features = None
        self.gate_tree_thresholds = None
        self.gate_tree_leaves = None
        
        self.up_tree_features = None
        self.up_tree_thresholds = None
        self.up_tree_leaves = None
        
        self.W_down_tree = None
        
        if self.tree_indices:
            self.tree_idx_t = torch.tensor(self.tree_indices, dtype=torch.long, device=device)
            
            gate_feats_list = []
            gate_thres_list = []
            gate_lvs_list = []
            
            up_feats_list = []
            up_thres_list = []
            up_lvs_list = []
            
            for j in self.tree_indices:
                # Cache stores gate and up tree data: cache[j] = (gate_data, up_data)
                gate_data, up_data = self.tree_cache[j]
                
                gate_feats_list.append(gate_data[0])
                gate_thres_list.append(gate_data[1])
                gate_lvs_list.append(gate_data[2])
                
                up_feats_list.append(up_data[0])
                up_thres_list.append(up_data[1])
                up_lvs_list.append(up_data[2])
                
            self.gate_tree_features = nn.Parameter(torch.tensor(gate_feats_list, dtype=torch.long, device=device), requires_grad=False)
            self.gate_tree_thresholds = nn.Parameter(torch.tensor(gate_thres_list, dtype=torch.float32, device=device), requires_grad=False)
            self.gate_tree_leaves = nn.Parameter(torch.tensor(gate_lvs_list, dtype=torch.float32, device=device), requires_grad=False)
            
            self.up_tree_features = nn.Parameter(torch.tensor(up_feats_list, dtype=torch.long, device=device), requires_grad=False)
            self.up_tree_thresholds = nn.Parameter(torch.tensor(up_thres_list, dtype=torch.float32, device=device), requires_grad=False)
            self.up_tree_leaves = nn.Parameter(torch.tensor(up_lvs_list, dtype=torch.float32, device=device), requires_grad=False)
            
            if isinstance(original_mlp, LlamaCompiledMLP) or isinstance(original_mlp, LlamaDualTreeCompiledMLP):
                W_down_tree_data = torch.zeros(2048, len(self.tree_indices), dtype=model_dtype, device=device)
                for col_idx, tree_neuron_idx in enumerate(self.tree_indices):
                    if tree_neuron_idx in original_mlp.tree_indices:
                        old_col_idx = original_mlp.tree_indices.index(tree_neuron_idx)
                        W_down_tree_data[:, col_idx] = original_mlp.W_down_tree.data[:, old_col_idx]
                    else:
                        old_col_idx = original_mlp.orig_indices.index(tree_neuron_idx)
                        W_down_tree_data[:, col_idx] = original_mlp.W_down_orig.data[:, old_col_idx]
                self.W_down_tree = nn.Parameter(W_down_tree_data, requires_grad=True)
            else:
                self.W_down_tree = nn.Parameter(original_mlp.down_proj.weight.data[:, self.tree_indices].clone(), requires_grad=True)
            
        self.act_fn = nn.SiLU()
        self.replaced_count = len(self.tree_indices)
        
    def forward(self, x):
        sh = x.shape
        x_flat = x.view(-1, 2048)
        N_samples = x_flat.size(0)
        
        out_flat = torch.zeros(N_samples, 2048, dtype=x.dtype, device=x.device)
        
        if self.orig_indices:
            z_gate = torch.matmul(x_flat, self.W_gate_orig.T)
            z_up = torch.matmul(x_flat, self.W_up_orig.T)
            a_orig = self.act_fn(z_gate) * z_up
            out_flat += torch.matmul(a_orig, self.W_down_orig.T)
            
        if self.tree_indices:
            N_trees = len(self.tree_indices)
            num_splits = (2 ** self.max_depth) - 1
            
            # 1. Gate Trees (SiLU gate activation)
            x_gate_features = x_flat[:, self.gate_tree_features.view(-1)].view(N_samples, N_trees, num_splits)
            decisions_gate = (x_gate_features > self.gate_tree_thresholds)
            curr_node_gate = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
            for _ in range(self.max_depth):
                dec_g = torch.gather(decisions_gate, 2, curr_node_gate.unsqueeze(2)).squeeze(2)
                curr_node_gate = torch.where(dec_g, 2 * curr_node_gate + 2, 2 * curr_node_gate + 1)
            leaf_idx_gate = curr_node_gate - num_splits
            a_gate = torch.gather(self.gate_tree_leaves, 1, leaf_idx_gate.T).T
            
            # 2. Value Trees (Up projection)
            x_up_features = x_flat[:, self.up_tree_features.view(-1)].view(N_samples, N_trees, num_splits)
            decisions_up = (x_up_features > self.up_tree_thresholds)
            curr_node_up = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
            for _ in range(self.max_depth):
                dec_u = torch.gather(decisions_up, 2, curr_node_up.unsqueeze(2)).squeeze(2)
                curr_node_up = torch.where(dec_u, 2 * curr_node_up + 2, 2 * curr_node_up + 1)
            leaf_idx_up = curr_node_up - num_splits
            a_up = torch.gather(self.up_tree_leaves, 1, leaf_idx_up.T).T
            
            # Combine activations via multiplication
            a_tree = (a_gate * a_up).to(x.dtype)
            out_flat += torch.matmul(a_tree, self.W_down_tree.T)
            
        return out_flat.view(sh[0], sh[1], 2048)


class FullyVectorizedLlamaMLP(nn.Module):
    def __init__(self, tree_features, tree_thresholds, tree_leaves_vectors, bias=None):
        super().__init__()
        self.tree_features = nn.Parameter(tree_features.to(torch.long), requires_grad=False)
        self.tree_thresholds = nn.Parameter(tree_thresholds.to(torch.float32), requires_grad=False)
        self.tree_leaves_vectors = nn.Parameter(tree_leaves_vectors.to(torch.float32), requires_grad=False)
        self.bias = nn.Parameter(bias.to(torch.float32) if bias is not None else torch.zeros(2048), requires_grad=False)
        self.max_depth = int(np.log2(tree_features.size(1) + 1))
        
    def forward(self, x):
        sh = x.shape
        x_flat = x.view(-1, 2048)
        N_samples = x_flat.size(0)
        
        N_trees = 8192
        num_splits = (2 ** self.max_depth) - 1
        num_leaves = 2 ** self.max_depth
        
        x_features = x_flat[:, self.tree_features.view(-1)].view(N_samples, N_trees, num_splits)
        decisions = (x_features > self.tree_thresholds)
        
        curr_node = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
        for _ in range(self.max_depth):
            dec = torch.gather(decisions, 2, curr_node.unsqueeze(2)).squeeze(2)
            curr_node = torch.where(dec, 2 * curr_node + 2, 2 * curr_node + 1)
            
        leaf_offset = num_splits
        leaf_idx = curr_node - leaf_offset
        
        out_flat = torch.zeros(N_samples, 2048, dtype=x.dtype, device=x.device)
        chunk_size = 128
        flat_leaves = self.tree_leaves_vectors.view(8192 * num_leaves, 2048)
        
        tree_idx = torch.arange(8192, device=x.device).unsqueeze(0)
        global_idx = tree_idx * num_leaves + leaf_idx
        
        for c in range(0, 8192, chunk_size):
            c_end = min(c + chunk_size, 8192)
            global_idx_chunk = global_idx[:, c:c_end]
            selected_vectors_chunk = flat_leaves[global_idx_chunk]
            out_flat += selected_vectors_chunk.sum(dim=1).to(x.dtype)
            
        out_flat += self.bias.to(x.dtype)
        return out_flat.view(sh[0], sh[1], 2048)


class QuantizedFullyVectorizedLlamaMLP(nn.Module):
    def __init__(self, tree_features, tree_thresholds, codebook, tree_leaves_indices, bias=None):
        super().__init__()
        self.tree_features = nn.Parameter(tree_features.to(torch.long), requires_grad=False)
        self.tree_thresholds = nn.Parameter(tree_thresholds.to(torch.float32), requires_grad=False)
        self.codebook = nn.Parameter(codebook.to(torch.float32), requires_grad=True)
        self.tree_leaves_indices = nn.Parameter(tree_leaves_indices.to(torch.long), requires_grad=False)
        self.bias = nn.Parameter(bias.to(torch.float32) if bias is not None else torch.zeros(2048), requires_grad=False)
        self.max_depth = int(np.log2(tree_features.size(1) + 1))
        
    def forward(self, x):
        sh = x.shape
        x_flat = x.view(-1, 2048)
        N_samples = x_flat.size(0)
        
        N_trees = 8192
        num_splits = (2 ** self.max_depth) - 1
        num_leaves = 2 ** self.max_depth
        
        x_features = x_flat[:, self.tree_features.view(-1)].view(N_samples, N_trees, num_splits)
        decisions = (x_features > self.tree_thresholds)
        
        curr_node = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
        for _ in range(self.max_depth):
            dec = torch.gather(decisions, 2, curr_node.unsqueeze(2)).squeeze(2)
            curr_node = torch.where(dec, 2 * curr_node + 2, 2 * curr_node + 1)
            
        leaf_offset = num_splits
        leaf_idx = curr_node - leaf_offset
        
        reconstructed_leaves = self.codebook[self.tree_leaves_indices]
        
        out_flat = torch.zeros(N_samples, 2048, dtype=x.dtype, device=x.device)
        chunk_size = 128
        flat_leaves = reconstructed_leaves.view(8192 * num_leaves, 2048)
        
        tree_idx = torch.arange(8192, device=x.device).unsqueeze(0)
        global_idx = tree_idx * num_leaves + leaf_idx
        
        for c in range(0, 8192, chunk_size):
            c_end = min(c + chunk_size, 8192)
            global_idx_chunk = global_idx[:, c:c_end]
            selected_vectors_chunk = flat_leaves[global_idx_chunk]
            out_flat += selected_vectors_chunk.sum(dim=1).to(x.dtype)
            
        out_flat += self.bias.to(x.dtype)
        return out_flat.view(sh[0], sh[1], 2048)


class LlamaDualTreeQuantizedMLP(nn.Module):
    def __init__(self, original_compiled_mlp, codebook, indices):
        super().__init__()
        self.layer_idx = original_compiled_mlp.layer_idx
        self.device = original_compiled_mlp.device
        self.model_dtype = original_compiled_mlp.model_dtype
        self.max_depth = original_compiled_mlp.max_depth
        
        self.tree_indices = original_compiled_mlp.tree_indices
        self.orig_indices = original_compiled_mlp.orig_indices
        
        # Original MLP weights
        self.W_gate_orig = original_compiled_mlp.W_gate_orig
        self.W_up_orig = original_compiled_mlp.W_up_orig
        self.W_down_orig = original_compiled_mlp.W_down_orig
        
        # Tree routing parameters (frozen)
        self.gate_tree_features = original_compiled_mlp.gate_tree_features
        self.gate_tree_thresholds = original_compiled_mlp.gate_tree_thresholds
        self.up_tree_features = original_compiled_mlp.up_tree_features
        self.up_tree_thresholds = original_compiled_mlp.up_tree_thresholds
        
        # Tree leaves (trainable)
        self.gate_tree_leaves = nn.Parameter(original_compiled_mlp.gate_tree_leaves.data.clone(), requires_grad=True)
        self.up_tree_leaves = nn.Parameter(original_compiled_mlp.up_tree_leaves.data.clone(), requires_grad=True)
        
        # Quantized W_down_tree
        self.codebook = nn.Parameter(codebook.clone(), requires_grad=True)
        self.indices = nn.Parameter(indices.clone(), requires_grad=False)
        
        self.act_fn = nn.SiLU()
        self.replaced_count = len(self.tree_indices)
        
    def forward(self, x):
        sh = x.shape
        x_flat = x.view(-1, 2048)
        N_samples = x_flat.size(0)
        
        out_flat = torch.zeros(N_samples, 2048, dtype=x.dtype, device=x.device)
        
        if self.orig_indices:
            z_gate = torch.matmul(x_flat, self.W_gate_orig.T)
            z_up = torch.matmul(x_flat, self.W_up_orig.T)
            a_orig = self.act_fn(z_gate) * z_up
            out_flat += torch.matmul(a_orig, self.W_down_orig.T)
            
        if self.tree_indices:
            N_trees = len(self.tree_indices)
            num_splits = (2 ** self.max_depth) - 1
            
            # 1. Gate routing
            x_gate_features = x_flat[:, self.gate_tree_features.view(-1)].view(N_samples, N_trees, num_splits)
            decisions_gate = (x_gate_features > self.gate_tree_thresholds)
            curr_node_gate = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
            for _ in range(self.max_depth):
                dec_g = torch.gather(decisions_gate, 2, curr_node_gate.unsqueeze(2)).squeeze(2)
                curr_node_gate = torch.where(dec_g, 2 * curr_node_gate + 2, 2 * curr_node_gate + 1)
            leaf_idx_gate = curr_node_gate - num_splits
            a_gate = torch.gather(self.gate_tree_leaves, 1, leaf_idx_gate.T).T
            
            # 2. Up routing
            x_up_features = x_flat[:, self.up_tree_features.view(-1)].view(N_samples, N_trees, num_splits)
            decisions_up = (x_up_features > self.up_tree_thresholds)
            curr_node_up = torch.zeros(N_samples, N_trees, dtype=torch.long, device=x.device)
            for _ in range(self.max_depth):
                dec_u = torch.gather(decisions_up, 2, curr_node_up.unsqueeze(2)).squeeze(2)
                curr_node_up = torch.where(dec_u, 2 * curr_node_up + 2, 2 * curr_node_up + 1)
            leaf_idx_up = curr_node_up - num_splits
            a_up = torch.gather(self.up_tree_leaves, 1, leaf_idx_up.T).T
            
            # 3. Combined activation
            a_tree = (a_gate * a_up).to(x.dtype)
            
            # 4. Reconstruct W_down_tree from codebook
            W_down_tree = self.codebook[self.indices] # shape: [8192, 2048]
            
            # 5. Output projection
            out_flat += torch.matmul(a_tree, W_down_tree)
            
        return out_flat.view(sh[0], sh[1], 2048)
