from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor



def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """

    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from .utils.toeknizer import Tokenizer
    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    from collections import defaultdict, Counter
    from utils.tokenization_util import run_serial_tokenization, run_parallel_tokenization, replace_best_pair_worker, count_pairs_worker
    from common import gpt2_bytes_to_unicode
    #https://medium.com/@hugmanskj/hands-on-bpe-byte-pair-encoding-%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%86%A0%ED%81%AC%EB%82%98%EC%9D%B4%EC%A0%80-%EA%B5%AC%ED%98%84-6bfef6f80f3b
    # Build GPT-2 byte-to-unicode mapping
    gpt2_vocab = gpt2_bytes_to_unicode()
    
    # Initialize vocabulary - this will be the final output vocab dict[int, bytes]
    vocab = {}
    token_to_id = {}
    idx = 0

    # Add special tokens to vocabulary first
    for token in special_tokens:
        vocab[idx] = token
        token_to_id[token] = idx
        idx += 1

    # Add all 256 possible byte values using GPT-2 encoding in the order they appear in gpt2_vocab
    # gpt2_vocab is a dict mapping byte values to unicode chars, we need to maintain this order
    for _, gpt2_char in gpt2_vocab.items():
        vocab[idx] = gpt2_char
        token_to_id[gpt2_char] = idx
        idx += 1

    # Read and tokenize the corpus
    pre_tokens = run_serial_tokenization(str(input_path))

    gpt2_style_pre_tokens = []
    for pre_token in pre_tokens:
        corpus = []
        for b in pre_token.encode('utf-8'):
            corpus.append(gpt2_vocab[b])
        gpt2_style_pre_tokens.append(corpus)

    pair2freq = Counter()
    pair2positions = defaultdict(set)  

    for token_idx, pre_token in enumerate(gpt2_style_pre_tokens):
        for i in range(len(pre_token) -1):
            pair = (pre_token[i], pre_token[i+1])
            pair2freq[pair] += 1 # pair : frequency
            pair2positions[pair].add((token_idx, i)) # pair : {(0, 0), (0, 1), ...} token_idx, token position

    merges = []
    while idx < vocab_size and pair2freq:
        best_pair = max(pair2freq, key=lambda pair: (pair2freq[pair], pair)) 
        positions = pair2positions[best_pair] # get best_pair positions
        merged_token = best_pair[0] + best_pair[1]

        vocab[idx] = merged_token
        token_to_id[merged_token] = idx
        idx += 1

        merges.append((best_pair[0], best_pair[1]))

        positions_by_pre_token = defaultdict(list)
        for pre_token_idx, pos in positions:
            positions_by_pre_token[pre_token_idx].append(pos) # best_pair_token_idx : token position
        
        del pair2freq[best_pair] # best pair 을 삭제한다.
        del pair2positions[best_pair] # best pair 을 삭제한다.

        pairs_to_remove = set() # remove pairs that will be affected by the merge
        pairs_to_add = []

        for pre_token_idx, pos_list in positions_by_pre_token.items():
            original_tokens = gpt2_style_pre_tokens[pre_token_idx]
            for pos in pos_list:
                if pos > 0:
                    left_pair = (original_tokens[pos - 1], original_tokens[pos])
                    pairs_to_remove.add((left_pair, pre_token_idx, pos - 1))
                
                if pos + 1 < len(original_tokens):
                    current_pair = (original_tokens[pos], original_tokens[pos+1])
                    pairs_to_remove.add((current_pair, pre_token_idx, pos))
                
                if pos + 2 < len(original_tokens):
                    right_pair = (original_tokens[pos+1], original_tokens[pos+2])
                    pairs_to_remove.add((right_pair, pre_token_idx, pos+1))

        # Remove old pairs
        for pair, pre_token_idx, pos in pairs_to_remove:
            if pair in pair2freq and (pre_token_idx, pos) in pair2positions[pair]:
                pair2freq[pair] -= 1
                pair2positions[pair].remove((pre_token_idx, pos))
                if pair2freq[pair] == 0:
                    del pair2freq[pair]
                    del pair2positions[pair]

        # Perform the actual merging
        for pre_token_idx, pos_list in positions_by_pre_token.items():
            pre_token = gpt2_style_pre_tokens[pre_token_idx]
            pos_array = sorted(pos_list, reverse=True)
            
            for pos in pos_array:
                if pos + 1 < len(pre_token):
                    pre_token[pos] = merged_token
                    del pre_token[pos + 1]

        # Add new pairs after merging
        for pre_token_idx, pos_list in positions_by_pre_token.items():
            tokens = gpt2_style_pre_tokens[pre_token_idx]
            sorted_positions = sorted(pos_list)
            
            for i, original_pos in enumerate(sorted_positions):
                adjusted_pos = original_pos - i
                
                if adjusted_pos > 0:
                    left_pair = (tokens[adjusted_pos-1], tokens[adjusted_pos])
                    pairs_to_add.append((left_pair, pre_token_idx, adjusted_pos-1))
                
                if adjusted_pos < len(tokens) - 1:
                    right_pair = (tokens[adjusted_pos], tokens[adjusted_pos+1])
                    pairs_to_add.append((right_pair, pre_token_idx, adjusted_pos))
        
        # Add new pairs to tracking
        for pair, pre_token_idx, pos in pairs_to_add:
            pair2freq[pair] += 1
            pair2positions[pair].add((pre_token_idx, pos))

    # Convert vocab values back to bytes for output
    final_vocab = {}
    for token_id, token_str in vocab.items():
        if isinstance(token_str, str):
            final_vocab[token_id] = token_str.encode('utf-8')
        else:
            final_vocab[token_id] = token_str

    return final_vocab, merges

def run_train_bpe_optimized(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    import time
    from collections import defaultdict
    from utils.tokenization_util import run_serial_tokenization, run_parallel_tokenization, replace_best_pair_worker, count_pairs_worker
    from common import gpt2_bytes_to_unicode
    #https://medium.com/@hugmanskj/hands-on-bpe-byte-pair-encoding-%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%86%A0%ED%81%AC%EB%82%98%EC%9D%B4%EC%A0%80-%EA%B5%AC%ED%98%84-6bfef6f80f3b
    total_start_time = time.time()
    print(f"Starting BPE training with vocab_size={vocab_size}")
    
    # Build GPT-2 byte-to-unicode mapping
    init_start = time.time()
    gpt2_vocab = gpt2_bytes_to_unicode()
    
    # Initialize vocabulary - this will be the final output vocab dict[int, bytes]
    vocab = {}
    # Also maintain a token-to-id mapping for building vocab
    token_to_id = {}
    idx = 0

    # Add special tokens to vocabulary first
    for token in special_tokens:
        vocab[idx] = token
        token_to_id[token] = idx
        idx += 1

    # Add all 256 possible byte values using GPT-2 encoding in the order they appear in gpt2_vocab
    # gpt2_vocab is a dict mapping byte values to unicode chars, we need to maintain this order
    for byte_val, gpt2_char in gpt2_vocab.items():
        vocab[idx] = gpt2_char
        token_to_id[gpt2_char] = idx
        idx += 1
    print(f"Initialization time: {time.time() - init_start:.4f}s")

    # Read and tokenize the corpus
    tokenization_start = time.time()
    docs = run_serial_tokenization(str(input_path))
    print(f"Corpus reading time: {time.time() - tokenization_start:.4f}s")
    
    corpus_process_start = time.time()
    # docs = [
    #     "low low low low low",
    #     "lower lower widest widest widest",
    #     "newest newest newest newest newest newest",
    # ]
    global_corpus = []
    for doc in docs:
        corpus = []
        doc_bytes = doc.encode('utf-8')
        for byte_val in doc_bytes:
            gpt2_encoded = gpt2_vocab[byte_val]
            corpus.append(gpt2_encoded)
        global_corpus.append(corpus)
    print(f"Corpus processing time: {time.time() - corpus_process_start:.4f}s")

    from collections import defaultdict, Counter
    initial_count_start = time.time()
    pair2freq = Counter()
    pair2positions = defaultdict(set)  # pair -> set of (doc_idx, pos)

    for doc_idx, tokens in enumerate(global_corpus):
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair2freq[pair] += 1
            pair2positions[pair].add((doc_idx, i))
    print(f"Initial pair counting time: {time.time() - initial_count_start:.4f}s")

    # Initialize merges list
    merges = []
    
    main_loop_start = time.time()
    iteration = 0
    
    while len(vocab) < vocab_size and pair2freq:
        iteration += 1
        iter_start = time.time()
        
        # find most freq pair
        find_best_start = time.time()
        best_pair = max(pair2freq, key=lambda pair: (pair2freq[pair], pair))
        positions = pair2positions[best_pair]
        merged_token = best_pair[0] + best_pair[1]
        vocab[idx] = merged_token
        token_to_id[merged_token] = idx
        merges.append((best_pair[0], best_pair[1]))
        idx += 1
        print(f"Iter {iteration}: Finding best pair took {time.time() - find_best_start:.4f}s")

        # For all positions where the pair occurs, merge them
        group_start = time.time()
        positions_by_doc = defaultdict(list)
        for doc_idx, pos in positions:
            positions_by_doc[doc_idx].append(pos)
        print(f"Iter {iteration}: Grouping positions took {time.time() - group_start:.4f}s")
        
        # remove old pair from stats
        cleanup_start = time.time()
        del pair2freq[best_pair]
        del pair2positions[best_pair]
        print(f"Iter {iteration}: Cleanup took {time.time() - cleanup_start:.4f}s")

        # Incremental position updates: only update pairs affected by the merge
        update_start = time.time()
        
        # Track pairs to remove and add (use sets to avoid duplicates)
        pairs_to_remove = set()
        pairs_to_add = []
        
        # Before merging, collect pairs that will be affected
        for doc_idx, pos_list in positions_by_doc.items():
            original_tokens = global_corpus[doc_idx]  # This is still the pre-merge version
            
            for pos in pos_list:
                # Remove old pairs around each merge position
                # Left neighbor pair: (pos-1, pos)
                if pos > 0 and pos < len(original_tokens):
                    left_pair = (original_tokens[pos-1], original_tokens[pos])
                    pairs_to_remove.add((left_pair, doc_idx, pos-1))
                
                # Current pair: (pos, pos+1) - this is the pair being merged
                if pos + 1 < len(original_tokens):
                    current_pair = (original_tokens[pos], original_tokens[pos+1])
                    pairs_to_remove.add((current_pair, doc_idx, pos))
                
                # Right neighbor pair: (pos+1, pos+2)
                if pos + 1 < len(original_tokens) and pos + 2 < len(original_tokens):
                    right_pair = (original_tokens[pos+1], original_tokens[pos+2])
                    pairs_to_remove.add((right_pair, doc_idx, pos+1))
        
        # Remove old pairs
        for pair, doc_idx, pos in pairs_to_remove:
            if pair in pair2freq and (doc_idx, pos) in pair2positions[pair]:
                pair2freq[pair] -= 1
                pair2positions[pair].remove((doc_idx, pos))
                if pair2freq[pair] == 0:
                    del pair2freq[pair]
                    del pair2positions[pair]
        
        # Now perform the actual merging (move this here from above)
        merge_start = time.time()
        import numpy as np
        for doc_idx, pos_list in positions_by_doc.items():
            tokens = global_corpus[doc_idx]
            if not pos_list:
                continue
                
            # Sort positions in reverse order to avoid index shifting issues
            pos_array = sorted(pos_list, reverse=True)
            
            # Apply merges
            for pos in pos_array:
                if pos + 1 < len(tokens):
                    tokens[pos] = merged_token
                    del tokens[pos + 1]  # Remove the second token
            
            global_corpus[doc_idx] = tokens
        print(f"Iter {iteration}: Merging took {time.time() - merge_start:.4f}s")
        
        # After merging, add new pairs
        for doc_idx, pos_list in positions_by_doc.items():
            tokens = global_corpus[doc_idx]
            
            # Calculate adjusted positions after merging
            sorted_positions = sorted(pos_list)
            for i, original_pos in enumerate(sorted_positions):
                # Account for deletions that happened before this position
                adjusted_pos = original_pos - i
                
                # Add new pairs around the merged position
                # Left neighbor: (adjusted_pos-1, adjusted_pos)
                if adjusted_pos > 0 and adjusted_pos < len(tokens):
                    left_pair = (tokens[adjusted_pos-1], tokens[adjusted_pos])
                    pairs_to_add.append((left_pair, doc_idx, adjusted_pos-1))
                
                # Right neighbor: (adjusted_pos, adjusted_pos+1)
                if adjusted_pos < len(tokens) - 1:
                    right_pair = (tokens[adjusted_pos], tokens[adjusted_pos+1])
                    pairs_to_add.append((right_pair, doc_idx, adjusted_pos))
        
        # Add new pairs
        for pair, doc_idx, pos in pairs_to_add:
            pair2freq[pair] += 1
            pair2positions[pair].add((doc_idx, pos))
        
        print(f"Iter {iteration}: Position updates took {time.time() - update_start:.4f}s")
        print(f"Iter {iteration}: Total iteration time: {time.time() - iter_start:.4f}s")
        print(f"Iter {iteration}: Remaining vocab to build: {vocab_size - len(vocab)}")
        print("---")
    
    print(f"Main loop total time: {time.time() - main_loop_start:.4f}s")

    # Convert back to int/bytes for output
    final_convert_start = time.time()
    print(f"Final conversion time: {time.time() - final_convert_start:.4f}s")
    print(f"Total BPE training time: {time.time() - total_start_time:.4f}s")
    
    return vocab, merges

if __name__ == "__main__":
    from common import FIXTURES_PATH
    input_path = FIXTURES_PATH / "corpus.en"
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print("Vocab:", vocab)
    print("Merges:", merges)