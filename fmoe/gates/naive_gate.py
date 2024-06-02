# r"""
# Naive gate
# """

# --------------------------------------------------------------------------------------------
# Gate Fraction Dropout

from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, dropout_prob=0.5):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)

        ###### Apply Dropout to Gate Values(Normal Dropout) ######
        # gate = self.dropout(gate)  # Apply dropout to gate values

        ######  My Design Dropout  ######
        #Property of Dropout => Pdrop = (exp_i / all_exp_grads_SUM)
        #gate_top_k_val = gate_top_k_val * Pdorp
        #
        
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]

        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).cuda())

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

# --------------------------------------------------------------------------------------------
# Expert Dropout

# from .base_gate import BaseGate

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class NaiveGate(BaseGate):
#     """
#     A naive gate implementation that defines the standard behavior of the gate
#     which determines which experts the tokens are going to.
#     Both the indices and the score, or confidence, are output to the parent
#     module.
#     The load-balance strategies are also designed to be implemented within the
#     `Gate` module.
#     """

#     def __init__(self, d_model, num_expert, world_size, top_k=2, dropout_prob=0.5):
#         super().__init__(num_expert, world_size)
#         self.gate = nn.Linear(d_model, self.tot_expert)
#         self.top_k = top_k
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, inp, return_all_scores=False):
#         """
#         The naive implementation simply calculates the top-k of a linear layer's
#         output.
#         """
#         gate = self.gate(inp)

#         # Apply dropout to gate values
#         gate = self.dropout(gate)

#         # Compute softmax probabilities after dropout
#         gate_probabilities = F.softmax(gate, dim=-1)

#         # Select top-k experts based on the modified gate probabilities
#         gate_top_k_val, gate_top_k_idx = torch.topk(
#             gate_probabilities, k=self.top_k, dim=-1, largest=True, sorted=False
#         )  # [.. x top_k]

#         # (BxL) x 1 x top_k
#         gate_score = gate_top_k_val.view(-1, self.top_k)

#         # dummy loss
#         self.set_loss(torch.zeros(1, requires_grad=True).cuda())

#         if return_all_scores:
#             return gate_top_k_idx, gate_score, gate
#         return gate_top_k_idx, gate_score