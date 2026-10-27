import torch
import torch.nn as nn
import numpy as np

class LlamaPiecewiseLinearMLP(nn.Module):
    def __init__(self, W_gate, W_up, W_down, tree_data, num_replace, device, dtype):
        super().__init__()
        # Store original weights as bfloat16 parameters or buffers
        self.W_gate = nn.Parameter(torch.tensor(W_gate, dtype=torch.bfloat16, device=device), requires_grad=False)
        self.W_up = nn.Parameter(torch.tensor(W_up, dtype=torch.bfloat16, device=device), requires_grad=False)
        self.W_down = nn.Parameter(torch.tensor(W_down, dtype=torch.bfloat16, device=device), requires_grad=False)
        
        self.num_replace = num_replace
        
        if num_replace > 0:
            # We have replaced neurons.
            # a_gate, b_gate, a_up, b_up: shape (num_replace, 15)
            # Max depth 3 tree has at most 15 nodes (indices 0 to 14)
            a_gate_init = np.zeros((num_replace, 15), dtype=np.float32)
            b_gate_init = np.zeros((num_replace, 15), dtype=np.float32)
            a_up_init = np.zeros((num_replace, 15), dtype=np.float32)
            b_up_init = np.zeros((num_replace, 15), dtype=np.float32)
            
            top_feats = []
            
            self.trees_gate = []
            self.trees_up = []
            
            for j in range(num_replace):
                tree_gate, tree_up, gate_coefs, up_coefs, top_idx = tree_data[j]
                self.trees_gate.append(tree_gate)
                self.trees_up.append(tree_up)
                top_feats.append(top_idx[0])
                
                # Fill coefficient arrays using leaf IDs
                for leaf, coef in gate_coefs.items():
                    if leaf < 15:
                        a_gate_init[j, leaf] = coef[0]
                        b_gate_init[j, leaf] = coef[1]
                for leaf, coef in up_coefs.items():
                    if leaf < 15:
                        a_up_init[j, leaf] = coef[0]
                        b_up_init[j, leaf] = coef[1]
            
            self.top_feats = torch.tensor(top_feats, dtype=torch.long, device=device)
            
            # Make the leaf coefficients trainable parameters in dtype
            self.a_gate = nn.Parameter(torch.tensor(a_gate_init, dtype=dtype, device=device), requires_grad=True)
            self.b_gate = nn.Parameter(torch.tensor(b_gate_init, dtype=dtype, device=device), requires_grad=True)
            self.a_up = nn.Parameter(torch.tensor(a_up_init, dtype=dtype, device=device), requires_grad=True)
            self.b_up = nn.Parameter(torch.tensor(b_up_init, dtype=dtype, device=device), requires_grad=True)
        else:
            self.register_parameter('a_gate', None)
            self.register_parameter('b_gate', None)
            self.register_parameter('a_up', None)
            self.register_parameter('b_up', None)

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        N = x_flat.shape[0]
        
        # 1. Compute original full MLP activations
        z_gate = x_flat @ self.W_gate.to(device).T
        z_up = x_flat @ self.W_up.to(device).T
        silu_gate = z_gate * torch.sigmoid(z_gate)
        y_orig = silu_gate * z_up # shape (N, 8192)
        
        if self.num_replace == 0:
            out_flat = y_orig @ self.W_down.to(device).T
            return out_flat.view(batch_size, seq_len, -1)
            
        # 2. Compute piecewise linear approximations for the replaced group
        # Extract top features
        # x_flat shape: (N, 2048), top_feats shape: (num_replace,)
        x_top1 = x_flat[:, self.top_feats] # shape (N, num_replace)
        
        # Compute leaf indices on CPU (since routing is discrete)
        x_np = x_flat.detach().cpu().to(torch.float32).numpy()
        leaf_gate = np.zeros((N, self.num_replace), dtype=np.int64)
        leaf_up = np.zeros((N, self.num_replace), dtype=np.int64)
        
        for j in range(self.num_replace):
            tree_gate = self.trees_gate[j]
            tree_up = self.trees_up[j]
            feat_idx = self.top_feats[j].item()
            # Feed the single top feature value to the tree
            v = x_np[:, feat_idx].reshape(-1, 1)
            leaf_gate[:, j] = tree_gate.apply(v)
            leaf_up[:, j] = tree_up.apply(v)
            
        # Convert to PyTorch tensors on GPU
        leaf_gate_pt = torch.tensor(leaf_gate, device=device) # shape (N, num_replace)
        leaf_up_pt = torch.tensor(leaf_up, device=device) # shape (N, num_replace)
        
        # Gather coefficients: shape (N, num_replace)
        # self.a_gate shape: (num_replace, 15)
        a_gate_gathered = torch.gather(self.a_gate.unsqueeze(0).expand(N, -1, -1), 2, leaf_gate_pt.unsqueeze(-1)).squeeze(-1)
        b_gate_gathered = torch.gather(self.b_gate.unsqueeze(0).expand(N, -1, -1), 2, leaf_gate_pt.unsqueeze(-1)).squeeze(-1)
        a_up_gathered = torch.gather(self.a_up.unsqueeze(0).expand(N, -1, -1), 2, leaf_up_pt.unsqueeze(-1)).squeeze(-1)
        b_up_gathered = torch.gather(self.b_up.unsqueeze(0).expand(N, -1, -1), 2, leaf_up_pt.unsqueeze(-1)).squeeze(-1)
        
        # Cast inputs and compute slope * input + intercept
        x_top1_cast = x_top1.to(dtype)
        gate_out = a_gate_gathered * x_top1_cast + b_gate_gathered
        up_out = a_up_gathered * x_top1_cast + b_up_gathered
        y_compiled = gate_out * up_out # shape (N, num_replace)
        
        # 3. Blend original and compiled activations
        # We replace the first `num_replace` neurons in the MLP output
        y_hybrid = y_orig.clone().to(dtype)
        y_hybrid[:, :self.num_replace] = y_compiled
        
        # 4. Project back
        out_flat = y_hybrid @ self.W_down.to(device).T
        return out_flat.view(batch_size, seq_len, -1)
