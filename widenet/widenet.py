import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
    ):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.router_layer = nn.Sequential(
            nn.Linear(
                inp_dim,
                num_experts,
            ),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor):
        return self.router_layer(x)


class ExpertAllocation(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        buffer_C: float,
        alpha: float = 0.01,
    ):
        super(ExpertAllocation, self).__init__()
        self.buffer_C = buffer_C
        self.alpha = alpha
        self.router = Router(inp_dim, num_experts)

    def forward(self, x):
        buffer_size = (x.shape[0] / self.num_experts) * self.buffer_C
        router_probs = self.router(x) # shape = (batch_size, num_experts)
        top_probs, top_idx = router_probs.topk(2, dim=-1) # shape = (batch_size, 2)
        routed_experts = torch.zeros_like(router_probs).scatter_(
            dim=-1,
            index=top_idx,
            src=torch.ones_like(top_probs),
        )

        aux_loss = 0
        if self.use_aux_loss:
            total_tokens = x.shape[0]
            f_i = torch.sum(routed_experts, dim=(0, 1)) * (1 / total_tokens)
            P_i = (torch.sum(router_probs, dim=(0, 1))) * (1 / total_tokens)

            aux_loss = self.alpha * self.num_experts * torch.sum((f_i * P_i))

        total_expert_allocation = routed_experts.cumsum(dim=0)
        expert_mask = (total_expert_allocation <= buffer_size).float()
        routed_experts = routed_experts * expert_mask
        routed_probs = routed_experts * routed_probs
        return routed_experts, routed_probs, top_idx, aux_loss

class MOELayer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        buffer_C: float,
    ):
        super(MOELayer, self).__init__()
        self.num_experts = num_experts
        self.buffer_C = buffer_C
        self.experts = nn.ModuleList(
            [nn.Linear(inp_dim, inp_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        routed_experts, routed_probs, top_idx, aux_loss = self.expert_allocation(x)
        active_tokens = (routed_experts.sum(dim=-1) > 0).view(-1) 
        routed_probs, top_idx = routed_probs.view(-1, 1), top_idx.view(-1)

        active_expert_idx = top_idx[active_tokens]
        active_x = x[active_tokens]
        active_out = torch.zeros_like(active_x)
        
        for i, expert in enumerate(self.experts):
            mask = active_expert_idx == i
            if mask.any():
                expert_out = expert(active_x[mask])
                active_out[mask] = expert_out
        
        active_out *= routed_probs[active_tokens]
        out = torch.zeros_like(x)
        out[active_tokens] = active_out
        out = out.view(x.shape)
        
        return out, aux_loss


class WideNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        inp_dim: int,
        num_layers: int,
        num_experts: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        buffer_C: float = 1.2,
    ):
        super(WideNet, self).__init__()
        self.moe_norms = nn.ModuleList([])
        self.attn_norms = nn.ModuleList([])

        for _ in range(num_layers):
            self.moe_norms.append(nn.LayerNorm(inp_dim))
            self.attn_norms.append(nn.LayerNorm(inp_dim))
        
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, inp_dim)
        self.attn_block = nn.MultiheadAttention(
            inp_dim, num_heads, attn_dropout, batch_first=True
        )
        self.moe_layer = MOELayer(inp_dim, num_experts, buffer_C)
    
    def forward(self, x):
        x = self.embedding(x)
        total_aux_loss = 0

        for i in range(self.num_layers):
            residual = x
            x = self.attn_norms[i](x)
            x = self.attn_block(x)
            x = x + residual

            residual = x
            x = self.moe_norms[i](x)
            x, aux_loss = self.moe_layer(x)
            total_aux_loss += aux_loss
            x = x + residual
        
        return x, total_aux_loss
