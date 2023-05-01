import torch
import torch.nn.functional as F


class AdvancedRepetitionPenaltyLogitsProcessor():
    def __init__(self, penalty: int, penalty_range: int, penalty_slope: int):
        self.penalty = penalty
        self.penalty_range = int(penalty_range)
        self.penalty_slope = penalty_slope

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        clipped_penalty_range = min(input_ids.shape[-1], self.penalty_range)

        if self.penalty != 1.0:
            if self.penalty_range > 0:
                if clipped_penalty_range < input_ids.shape[1]:
                    input_ids = input_ids[..., -clipped_penalty_range:]

                if self.penalty_slope != 0:
                    _penalty = (torch.arange(self.penalty_range, dtype=scores.dtype, device=scores.device)/(self.penalty_range - 1)) * 2. - 1
                    _penalty = (self.penalty_slope * _penalty) / (1 + torch.abs(_penalty) * (self.penalty_slope - 1))
                    _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (self.penalty - 1)
                    self.penalty = _penalty[..., -clipped_penalty_range:]

            score = torch.gather(scores, 1, input_ids)
            score = torch.where(score <= 0, score * self.penalty, score / self.penalty)
            scores.scatter_(1, input_ids, score)

        return scores


class TailFreeLogitsWarper():

    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper():
    '''
    Typical sampling, described in https://arxiv.org/pdf/2202.00666.pdf
    '''

    def __init__(self, typical: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        typical = float(typical)
        if typical < 0 or typical > 1.0:
            raise ValueError(f"`typical` has to be a float >= 0 and <= 1, but is {typical}")
        self.typical = typical
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores

        # Compute softmax probabilities and the natural logarithms of them
        probs = scores.softmax(dim=-1)
        log_probs = probs.log()

        # Compute the negative of entropy, which is the sum of p*ln(p) for all p
        # in the set of softmax probabilities of the logits
        neg_entropy = (probs * log_probs).nansum(dim=-1, keepdim=True)

        # Determine absolute difference between the negative entropy and the
        # log probabilities
        entropy_deviation = (neg_entropy - log_probs).abs()

        # Keep certain tokens such that the sum of the entropy_deviation of the
        # kept tokens is the smallest possible value such that the sum of the
        # softmax probabilities of the kept tokens is at least the threshold
        # value (by sorting the tokens in ascending order of entropy_deviation
        # and then keeping the smallest possible number of tokens from the
        # beginning such that sum of softmax probabilities is at or above the
        # threshold)
        _, sorted_indices = torch.sort(entropy_deviation)
        sorted_logits = probs.gather(-1, sorted_indices)
        sorted_indices_to_remove = sorted_logits.cumsum(dim=-1) >= self.typical
        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)

        min_tokens_to_keep = max(self.min_tokens_to_keep, 1)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper():
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class MinLengthLogitsProcessor():
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.
    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class TemperatureLogitsWarper():
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsWarper():
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores[indices_to_remove] = self.filter_value
        return scores


class TopKLogitsWarper():
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores[indices_to_remove] = self.filter_value
        return scores
