import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        is_greedy = temperatures == 0
        # temperature 0 means greedy; divide those rows by 1 so they don't become inf/nan
        logits = logits.div_(torch.where(is_greedy, 1.0, temperatures).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(is_greedy, greedy_tokens, sample_tokens)
