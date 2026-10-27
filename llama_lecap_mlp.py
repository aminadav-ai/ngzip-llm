import torch
import torch.nn as nn

class LlamaLecapMLP(nn.Module):
    def __init__(self, W_gate, W_up, W_down, device, dtype):
        super().__init__()
        # Store W_gate and W_up as trainable parameters
        self.W_gate = nn.Parameter(torch.tensor(W_gate, dtype=dtype, device=device), requires_grad=True)
        self.W_up = nn.Parameter(torch.tensor(W_up, dtype=dtype, device=device), requires_grad=True)
        self.W_down = nn.Parameter(torch.tensor(W_down, dtype=dtype, device=device), requires_grad=False)
        
        # Initialize binary masks based on parameter shape
        intermediate_size, hidden_dim = W_gate.shape
        self.register_buffer('gate_mask', torch.ones(intermediate_size, hidden_dim, dtype=torch.bool, device=device))
        self.register_buffer('up_mask', torch.ones(intermediate_size, hidden_dim, dtype=torch.bool, device=device))
        
        # Register backward hooks to zero out gradients for masked-out weights
        self.W_gate.register_hook(lambda grad: grad * self.gate_mask)
        self.W_up.register_hook(lambda grad: grad * self.up_mask)

    def enforce_masks(self):
        # Explicitly zero out any weights that are outside the masks
        with torch.no_grad():
            self.W_gate.data.mul_(self.gate_mask)
            self.W_up.data.mul_(self.up_mask)

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Ensure weights outside the masks are exactly zero
        self.enforce_masks()
        
        # Compute forward pass
        z_gate = x_flat @ self.W_gate.T
        z_up = x_flat @ self.W_up.T
        silu_gate = z_gate * torch.sigmoid(z_gate)
        y = silu_gate * z_up # shape (N, intermediate_size)
        
        out_flat = y @ self.W_down.T
        return out_flat.view(batch_size, seq_len, -1)
