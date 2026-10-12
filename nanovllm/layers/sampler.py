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


def verify_drafts(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_drafts: torch.Tensor,
    temperatures: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    chunk_rows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculative decoding: decide how many guessed tokens to keep, then pick one more token.

    S sequences, each with K guess slots (the first num_drafts[s] are real, the rest padding).
    logits:          [S*(K+1), V]. Row s*(K+1)+j is what the model predicts after reading the
                     last real token and guesses 1..j, i.e. the model's opinion about guess j+1.
    draft_token_ids: [S, K]
    num_drafts:      [S]
    temperatures:    [S], 0 means greedy
    draft_probs:     [S, K, V] the distribution q each guess was sampled from (draft model),
                     or None for guesses that are not sampled (n-gram): q is 1 on the guess.
    Returns (num_accepted [S], next_token [S]). Sequence s keeps draft_token_ids[s, :num_accepted[s]]
    followed by next_token[s].

    Greedy: keep guesses while they equal the model's top-1; next_token is the model's top-1 at the
    first mismatch (or after the last guess). Output is exactly what plain greedy decoding gives.
    Random: keep guess d with probability min(1, p(d)/q(d)); at the first rejection sample from
    max(p - q, 0) renormalized; if all are kept, sample a bonus token from p. Standard speculative
    sampling: the output has exactly the same distribution as sampling from p token by token.
    """
    S, K = draft_token_ids.shape
    V = logits.size(-1)
    device = logits.device
    logits = logits.view(S, K + 1, V)
    rows = torch.arange(S, device=device)
    is_greedy = temperatures == 0
    temps = torch.where(is_greedy, 1.0, temperatures).float()    # avoid dividing by 0 on greedy rows
    any_random = not bool(is_greedy.all())

    top1 = logits.argmax(dim=-1)    # [S, K+1]
    accepted = top1[:, :K] == draft_token_ids
    if any_random and K > 0:
        # p(d) = exp(logit_d / T - logsumexp(logits / T)). Work in row chunks so we never hold the
        # whole [S*(K+1), V] table in float32 (with 512 requests and k=4 that alone is ~1.5GB).
        flat = logits.view(S * (K + 1), V)
        row_temps = temps.repeat_interleave(K + 1)
        lse = torch.empty(S * (K + 1), device=device, dtype=torch.float32)
        for i in range(0, S * (K + 1), chunk_rows):
            lse[i:i + chunk_rows] = torch.logsumexp(flat[i:i + chunk_rows].float() / row_temps[i:i + chunk_rows, None], dim=-1)
        lse = lse.view(S, K + 1)[:, :K]
        draft_logits = logits[:, :K].gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
        p_draft = torch.exp(draft_logits / temps[:, None] - lse)
        if draft_probs is None:
            q_draft = torch.ones_like(p_draft)
        else:
            q_draft = draft_probs.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1).float()
        u = torch.rand(S, K, device=device, dtype=torch.float32)
        accepted = torch.where(is_greedy[:, None], accepted, u * q_draft < p_draft)    # P(accept) = min(1, p/q)
    accepted &= torch.arange(K, device=device) < num_drafts.unsqueeze(1)
    num_accepted = accepted.int().cumprod(dim=1).sum(dim=1)    # length of the unbroken run of accepts

    next_token = top1[rows, num_accepted]
    if any_random:
        p = torch.softmax(logits[rows, num_accepted].float() / temps[:, None], dim=-1)    # [S, V]
        if K > 0:
            rejected = num_accepted < num_drafts
            j = num_accepted.clamp(max=K - 1)    # the rejected guess (only meaningful where rejected)
            if draft_probs is None:
                # q is all on the guess, so max(p - q, 0) is p with the guessed token removed
                d = draft_token_ids[rows, j].unsqueeze(1)
                p.scatter_(1, d, torch.where(rejected[:, None], 0.0, p.gather(1, d)))
            else:
                p = torch.where(rejected[:, None], (p - draft_probs[rows, j].float()).clamp_min_(0), p)
        # p / Exp(1) then argmax samples proportionally to p (no need to renormalize)
        sampled = p.div_(torch.empty_like(p).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        next_token = torch.where(is_greedy, next_token, sampled)
    return num_accepted, next_token
