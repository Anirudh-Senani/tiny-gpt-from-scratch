"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    return {key:val for val, key in zip(range(len(vocab)),vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # TODO: build an int-to-string lookup from the vocab list
    return {key:val for key, val in zip(range(len(vocab)),vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    return [stoi[ch] for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    return "".join([itos[ind] for ind in ids])

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.asarray(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    # TODO: allocate a (rows, cols) array of zeros and return it
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    # TODO: build a seeded RNG and draw a (rows, cols) uniform sample in [0, 1).
    rng = np.random.default_rng(seed)
    return rng.random((rows,cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i,j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    # TODO: return the i-th row of arr as a 1D array of shape (C,)
    return arr[i,:]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:,j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a+b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix + vector[None,:]

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    # TODO: collapse every element of arr into a single scalar total
    return arr.sum()

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    # TODO: reduce the row dimension of arr so the result has shape (C,).
    return arr.sum(axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return arr.sum(axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    # TODO: compute the maximum value of arr along the given axis
    return arr.max(axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    # TODO: compute the matrix product of a and b
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # TODO: sum along the given axis preserving the reduced dim as size 1
    return arr.sum(axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    # TODO: exponentiate the logits, then divide by their total sum
    exp_logits = np.exp(logits)
    return exp_logits/exp_logits.sum()

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    # TODO: exponentiate large_value via array_exp and report whether it is inf.
    exp_val = np.exp(large_value)
    return dict(
        naive_exp=exp_val,
        overflowed=np.isinf(exp_val)
    )

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    # TODO: subtract the max before exponentiating, then normalize.
    exp_logits = np.exp(logits - logits.max())
    return exp_logits/exp_logits.sum()

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    # TODO: turn each row of logits into a probability distribution without overflowing
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp_logits/exp_logits.sum(axis=-1, keepdims=True)

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    # TODO: validate that text_blob is a non-empty str and return it as the corpus string
    if not isinstance(text_blob, str):
        raise TypeError
    elif not text_blob:
        raise ValueError
    
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    # TODO: map every character in text through stoi and return as a 1D int64 array
    return np.array([stoi[ch] for ch in text])

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    # TODO: compute the integer split index from n and train_frac
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return data[:split_idx], data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    # TODO: return an integer block size, at least 1, derived from default_size
    return max(default_size, 1)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    # TODO: extract a single input window of length block_size starting at index i
    return data[i:i+block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i+1:i+1+block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    # TODO: sample batch_size offsets in the valid range for a (block_size+1)-window.
    return rng.integers(0, data_len-block_size, batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    # TODO: for each offset, take a length-block_size slice of data and stack them as rows
    return np.vstack([slice_x_at_offset(data, i, block_size) for i in offsets])

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    return np.vstack([slice_y_at_offset(data, i, block_size) for i in offsets])

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    return stack_x_batch(data, offsets, block_size), stack_y_batch(data, offsets, block_size)

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    # TODO: return a (vocab_size, vocab_size) integer array of zeros.
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    # TODO: walk consecutive (current, next) pairs in data and add 1 to the matching cell
    for i in range(len(data)-1):
        n_matrix[data[i], data[i+1]] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # TODO: allocate counts, then scatter-add 1 at each (data[:-1], data[1:]) pair
    n_matrix = allocate_count_matrix(vocab_size)
    np.add.at(n_matrix, (data[:-1], data[1:]), 1)
    return n_matrix
    # return loop_fill_counts(n_matrix, data)

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # TODO: apply +1 Laplace smoothing to the bigram count matrix
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    # TODO: compute per-row sums of the count matrix as a column vector for normalization.
    return sum_keepdims(n_matrix, axis=-1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    # TODO: divide each row of n_matrix by its row sum to produce probabilities
    return n_matrix/row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    # TODO: draw one categorical sample from the row of p_matrix at current_id
    return rng.choice(p_matrix.shape[1], p=p_matrix[current_id])

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    tokens = [start_id]
    for _ in range(length-1):
        tokens.append(sample_next_token(p_matrix, tokens[-1], rng))
    return np.asarray(tokens)

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    return "".join(itos[ind] for ind in ids)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    return np.log(p_matrix[current_id, next_id])

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    return (-np.log(p_matrix[data[:-1],data[1:]])).sum()

# Step 56 - average_nll
def average_nll(p_matrix, data):
    # TODO: return mean negative log likelihood per bigram over consecutive pairs in data.
    nll_sum = sum_negative_log_probs(p_matrix, data)
    return nll_sum/(max(data.shape[0]-1,1))

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal((vocab_size,vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    # TODO: allocate an (N, vocab_size) zero matrix and set one 1 per row at ids[i]
    one_hot = make_2d_zeros(len(ids), vocab_size)
    one_hot[np.arange(one_hot.shape[0]), ids] = 1.0
    return one_hot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    # TODO: compute logits for the neural bigram model as the matrix product of one-hot inputs and W.
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    # TODO: compute logits two ways and return both in a dict
    vocab_size = w.shape[0]
    one_hot = one_hot_encode_batch(ids, vocab_size)
    return dict(
        onehot_result=forward_logits_onehot(one_hot, w),
        index_result=w[ids]
    )

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    # TODO: return the logits for a batch of token ids by direct row lookup into W.
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    # TODO: pick out the probability assigned to the correct next token for each batch row
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    # TODO: gather correct-token probs, take log, average the negatives
    correct_token_probs = gather_correct_token_probs(probs, targets)
    return (-np.log(correct_token_probs)).mean()

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    # TODO: return a short written derivation ending in dL/dlogits = (probs - onehot(targets)) / B
    return """dL/dL = 1; dL/dy = -dL/dL; dL/dtargets = (dL/dL * dL/sum(softmax(logits)))/B; dL/dlogits = (probs - onehot(targets)) / B"""

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    # TODO: return dL/dlogits of shape (B, V) averaged over the batch.
    batch_size, vocab_size = probs.shape
    onehot_target = one_hot_encode_batch(targets, vocab_size)
    return (probs - onehot_target)/batch_size

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    # TODO: return a fixed multi-line string describing the scatter-add gradient.
    return """Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\nShapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\nChain rule: dL/dW = O.T @ dlogits, shape (V, D).\nSince O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\nRow v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\nImplementation: scatter-add dlogits rows into dW at indices ids."""

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    # TODO: build a (vocab_size, vocab_size) dW and accumulate dlogits[b] into row ids[b].
    return one_hot_encode_batch(ids, vocab_size).T @ dlogits

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    # TODO: subtract the scaled gradient from the weights and return the new matrix
    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    # TODO: chain the upstream forward/loss/backward/update helpers into one step
    vocab_size = w.shape[0]

    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)

    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, vocab_size)

    new_w = sgd_update_w(w, dw, learning_rate)

    return dict(
        w=new_w,
        loss=loss
    )

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    # TODO: repeatedly sample a batch, run one training step, and log loss every log_every steps
    loss_history = []
    rng = np.random.default_rng()
    for step in range(num_steps):
        ids, targets = get_batch(data, block_size, batch_size, rng)
        train_step = run_one_training_step(w, ids.flatten(), targets.flatten(), learning_rate)
        w = train_step['w']
        if step % log_every == 0:
            loss_history.append(train_step['loss'])

    return dict(
        w=w,
        loss_history=loss_history
    )

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    ids = [start_id]
    rng = np.random.default_rng()
    token_id = start_id
    for step in range(num_tokens):
        logits = forward_logits_lookup(w, ids)
        probs = logits_to_probs_rowwise(logits)
        token_id = sample_next_token(probs, step, rng)
        ids.append(token_id)
    return decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    return dict(
        y=x @ w,
        cache=dict(x=x, w=w)
    )

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    # TODO: return a multi-line string with the derivation and shape check
    return """Y = X @ W
dL/dX = dY @ W.T
shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"""

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    # TODO: return notes that include the final identity dL/dW = X.T @ dY
    return """Y = X @ W
dL/dW = X.T @ dY
shapes: X (B, D_in), W (D_in, D_out), dY (B, D_out) -> dL/dW (B, D_out)"""

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    return dy @ cache['w'].T

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    return cache['x'].T @ dy

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    # TODO: add b to each row of x and cache b's shape for the backward pass
    return dict(
        y=x+b,
        cache=dict(b_shape=b.shape)
    )

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    # TODO: sum the upstream gradient over the batch dimension to get db of shape (D,)
    return dy.sum(axis=0)

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    return dict(
        y=np.maximum(x, 0.0),
        cache=dict(x=x)
    )

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    # TODO: return dx with gradient zeroed where the cached input was non-positive.
    return np.where(cache['x']>0.0, dy, 0.0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    return compute_dlogits(probs, targets)

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    # TODO: compute the per-row mean of x, preserving the reduced axis as size 1
    return sum_keepdims(x, -1)/x.shape[-1]

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    # TODO: compute per-row variance using mean and return a (B, 1) array
    return (sum_keepdims((x-mean)**2, -1))/x.shape[-1]

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    return (x-mean)/np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    norm = layernorm_forward_normalize(x, mean, var, eps)
    y = vector_matrix_broadcast_add(elementwise_multiply(norm, gamma), beta)
    return dict(
        y=y,
        cache=dict(x=x, x_hat=norm, mean=mean, var=var, gamma=gamma, eps=eps)
    )

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    # TODO: compute the gradient contribution of the subtract-mean op
    return dy - (sum_keepdims(dy, axis=-1)/dy.shape[-1])

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    return dy/((cache['var']+cache['eps'])**0.5)

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    # TODO: chain rule back through affine, divide-by-std, and subtract-mean.
    dxhat = cache['gamma'] * dy
    dmean = layernorm_backward_subtract_mean(dxhat, cache)
    variance_correction = cache['x_hat'] * sum_keepdims(dxhat * cache['x_hat'], -1)/dxhat.shape[-1]
    dx = layernorm_backward_divide_std(dmean - variance_correction, cache)
    return dict(
        dx=dx,
        dgamma=(dy*cache['x_hat']).sum(axis=0),
        dbeta=(dy).sum(axis=0)
    )

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    return layernorm_backward_full(d_out, cache)

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    # TODO: return a (vocab_size, d_model) array of small random values controlled by scale
    w = np.random.standard_normal((vocab_size, d_model))
    return w * scale

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    # TODO: look up the embedding row for each token id and build the cache
    return embedding_matrix[token_ids], dict(token_ids=token_ids, vocab_size=embedding_matrix.shape[0])

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    dE = np.zeros((cache['vocab_size'], d_out.shape[-1]))
    np.add.at(dE, cache['token_ids'], d_out)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    pe = make_2d_random(block_size, d_model, None)
    pe = scale_w_small(pe, scale)
    return pe

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    return positional_matrix[:seq_len, :]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    return token_emb + pos_emb[None,:,:]

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    # TODO: route d_out to both branches, reducing over the batch axis for pos_emb.
    return dict(
        d_token_emb=d_out,
        d_pos_emb=d_out.sum(axis=0)
    )

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    # TODO: return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)
    Wq = make_2d_random(d_model, d_head, 0)
    Wk = make_2d_random(d_model, d_head, 1)
    Wv = make_2d_random(d_model, d_head, 2)
    Wq = scale_w_small(Wq, scale)
    Wk = scale_w_small(Wk, scale)
    Wv = scale_w_small(Wv, scale)

    return dict(
        Wq=Wq,
        Wk=Wk,
        Wv=Wv
    )

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    # TODO: project x into the query space using w_q
    return x @ w_q

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    # TODO: project the (B, T, d_model) input through w_k to produce (B, T, d_head) keys.
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    # TODO: project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)
    return x @ w_v

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    # TODO: compute raw attention scores Q @ K^T per batch element
    return q @ np.transpose(k, axes=(0,2,1))

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    return scores/(d_head**0.5)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    # TODO: build a (T, T) boolean mask where True marks allowed (query, key) pairs
    return np.tril(np.full((seq_len, seq_len), True))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    # TODO: return a (B,T,T) array where positions with causal_mask False are -inf...
    return scaled_scores + np.where(causal_mask, 0.0, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    # TODO: apply numerically stable softmax along the last axis of masked_scores
    exp_scores = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    return exp_scores/exp_scores.sum(axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    # TODO: mix the value vectors using the attention weights
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    # TODO: return attn_out projected through w_o to shape (B, T, d_model)
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    # TODO: backprop through proj = attn_out @ w_o, return gradients for attn_out and w_o
    return dict(
        d_attn_out=d_proj @ cache['w_o'].T,
        dw_o=(cache['attn_out'].transpose((0,2,1)) @ d_proj).sum(axis=0)
    )

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    # TODO: backprop through out = attn @ V to obtain gradients for attn and V.
    return dict(
        d_attn=d_attn_out @ cache['v'].transpose((0,2,1)),
        d_v=cache['attn'].transpose((0,2,1)) @ d_attn_out
    )

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    # TODO: propagate the softmax Jacobian per row and zero out masked positions.
    d_scores = cache['attn']*(d_attn - (cache['attn']*d_attn).sum(axis=-1, keepdims=True))
    return d_scores * cache['causal_mask'][None,:,:]

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    # TODO: propagate d_scaled_scores back through the sqrt(d_head) scaling
    return d_scaled_scores/(d_head**0.5)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    # TODO: backprop scores = Q @ K^T to obtain gradients for Q and K
    return dict(
        d_q=d_scores @ cache['k'],
        d_k=d_scores.transpose((0,2,1)) @ cache['q']
    )

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    # TODO: backprop through Q=x@Wq, K=x@Wk, V=x@Wv to get dx and dw_q, dw_k, dw_v.
    return dict(
        dx=d_q @ cache['w_q'].T + d_k @ cache['w_k'].T + d_v @ cache['w_v'].T,
        dw_q = (cache['x'].transpose((0,2,1)) @ d_q).sum(axis=0),
        dw_k = (cache['x'].transpose((0,2,1)) @ d_k).sum(axis=0),
        dw_v = (cache['x'].transpose((0,2,1)) @ d_v).sum(axis=0)
    )

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    # TODO: split d_model into n_heads equal-sized d_head chunks and return the config dict
    if d_model%n_heads:
        raise ValueError
    return dict(
        n_heads=n_heads,
        d_head=d_model//n_heads,
        d_model=d_model
    )

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    # TODO: build a dict with keys 'Wq', 'Wk', 'Wv', each a scaled (d_model, d_model) random matrix
    Wq = make_2d_random(d_model, d_model, 0)
    Wk = make_2d_random(d_model, d_model, 1)
    Wv = make_2d_random(d_model, d_model, 2)

    Wq = scale_w_small(Wq, scale)
    Wk = scale_w_small(Wk, scale)
    Wv = scale_w_small(Wv, scale)

    return dict(
        Wq=Wq,
        Wk=Wk,
        Wv=Wv
    )

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    # TODO: build a (d_model, d_model) random matrix and scale it down by `scale`.
    Wo = make_2d_random(d_model, d_model, 0)
    Wo = scale_w_small(Wo, scale)
    return Wo

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    # TODO: split the last dimension of x into n_heads chunks of size d_head
    B, T, _ = x.shape
    return x.reshape((B, T, n_heads, d_head))

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    # TODO: move the heads axis in front of the time axis
    return x_heads.transpose((0,2,1,3))

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    # TODO: return the number of attention heads stored in the multi-head config dict.
    return config['n_heads']

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    # TODO: return the sequence length T from the (B, T, d_model) tensor.
    return x.shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    # TODO: return the per-head dimension d_head for multi-head attention.
    if d_model%n_heads:
        raise ValueError
    return d_model//n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    # TODO: mask future positions then row-wise softmax over the last axis
    masked_scores = scores + np.where(mask, 0.0, -np.inf)[None, None, :, :]
    exp_scores = np.exp(masked_scores - masked_scores.max(axis=-1, keepdims=True))
    return exp_scores/exp_scores.sum(axis=-1, keepdims=True)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    # TODO: combine attention weights with values across heads
    return weights @ v_heads

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    # TODO: move the heads axis back so the result has shape (B, T, n_heads, d_head).
    return x_heads.transpose((0,2,1,3))

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    # TODO: read the sequence-length dimension from x_heads_back's shape
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    # TODO: collapse the last two axes into a single d_model axis
    B, T, N, H = x_heads_back.shape
    return x_heads_back.reshape((B, T, N*H))

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
        merged: (B, T, d_model)
        w_out:  (d_model, d_model)
        b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    # TODO: project merged through w_out, add b_out, and stash inputs in the cache.
    out_dict = linear_forward(merged, w_out)
    out_dict = bias_add_forward(out_dict['y'], b_out)

    return dict(
      out=out_dict['y'],
      cache=dict(merged=merged,w_out=w_out)
    )

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    d_heads = transpose_heads_to_front(reshape_to_heads(d_merged, shape_info['n_heads'], shape_info['d_head']))
    return d_heads

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    # TODO: apply the first FFN linear that expands d_model to d_ff
    out_dict = linear_forward(x, w1)
    out_dict = bias_add_forward(out_dict['y'], b1)
    return dict(
        h1=out_dict['y'],
        cache=dict(x=x,w1=w1)
    )

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    # TODO: apply ReLU activation in the FFN hidden layer and cache h1
    return np.maximum(h1, 0.0),dict(h1=h1)

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    # TODO: project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2, return h2 and cache
    out_dict = linear_forward(a1, w2)
    out_dict = bias_add_forward(out_dict['y'],b2)
    return dict(
        h2=out_dict['y'],
        cache=dict(a1=a1, w2=w2)
    )

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    # TODO: route d_out back through linear2, ReLU, and linear1 to get input and param grads
    db2 = d_out.sum(axis=(0,1))
    dw2 = (cache['a1'].transpose((0,2,1)) @ d_out).sum(axis=0)
    da1 = d_out @ cache['w2'].T
    dh1 = np.where(cache['h1']<=0.0, 0.0, da1)
    db1 = dh1.sum(axis=(0,1))
    dw1 = (cache['x'].transpose((0,2,1)) @ dh1).sum(axis=0)
    dx = dh1 @ cache['w1'].T

    return dict(
        dx=dx,
        dw1=dw1,
        db1=db1,
        dw2=dw2,
        db2=db2
    )

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    # TODO: add the sublayer output to its input to form a residual connection.
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    # TODO: route the upstream gradient to both branches of the residual add.
    return d_y.copy(), d_y

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # TODO: apply LayerNorm to x, run sublayer_fn on the result, then residual-add back to x.
    if 'eps' not in ln_params:
        ln_params['eps'] = 1e-5
    pre_norm = layernorm_forward_affine(x, **ln_params)
    sublayer = sublayer_fn(pre_norm['y'], sublayer_params)
    return dict(
        y=x+sublayer['y'],
        cache=dict(x=x,ln_cache=pre_norm['cache'],sublayer_cache=sublayer['cache'])
    )

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    # TODO: compose pre-LN attention sublayer then pre-LN FFN sublayer with residuals
    B, T, d_model = x.shape
    def attn_fn(x, attn_params):
        q = x @ attn_params['Wq']
        k = x @ attn_params['Wk']
        v = x @ attn_params['Wv']
        d_head = compute_d_head(d_model,attn_params['n_heads'])
        scale = 1.0/(d_head**0.5)

        q = transpose_heads_to_front(reshape_to_heads(q, attn_params['n_heads'], d_head))
        k = transpose_heads_to_front(reshape_to_heads(k, attn_params['n_heads'], d_head))
        v = transpose_heads_to_front(reshape_to_heads(v, attn_params['n_heads'], d_head))
        mask = build_causal_mask(T)

        scores = (q @ k.transpose((0,1,3,2)))*scale
        scores = multihead_masked_softmax_scores(scores, mask)
        scores = merge_heads_to_d_model(transpose_heads_to_back(multihead_weighted_sum(scores, v)))
        out = multihead_output_projection_forward(scores, attn_params['Wo'], attn_params['bo'])['out']

        return dict(
            y=out,
            cache=dict(
                x=x,
                q=q,
                k=k,
                v=v,
                Wq=attn_params['Wq'],
                Wk=attn_params['Wk'],
                Wv=attn_params['Wv'],
                Wo=attn_params['Wo'],
                merged=scores,
                mask=mask,
                scale=scale
            )
        )

    def ffn_fn(x, ffn_params):
        h1 = ffn_linear_one_forward(x, ffn_params['w1'], ffn_params['b1'])['h1']
        a1 = ffn_activation_forward(h1)[0]
        out = ffn_linear_two_forward(a1, ffn_params['w2'], ffn_params['b2'])['h2']
        return dict(
            y=out,
            cache=dict(
                x=x,
                w1=ffn_params['w1'],
                h1=h1,
                a1=a1,
                w2=ffn_params['w2']
            )
        )

    attn_out = pre_layernorm_sublayer_forward(x, block_params['ln1'], attn_fn, block_params['attn'])
    # h1 = x+attn_out['y']
    ffn_out = pre_layernorm_sublayer_forward(attn_out['y'], block_params['ln2'], ffn_fn, block_params['ffn'])
    return dict(
        y=ffn_out['y'],
        cache=dict(attn_branch=attn_out['cache'], ffn_branch=ffn_out['cache'])
    )

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys 'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested dict
        with keys 'ln1', 'ln2', 'attn', 'ffn' mirroring block_params.
    """
    # Tip: recover x from cache['attn_branch']['x'] and call _complete_block_cache(x, block_params)
    # to guarantee every field the backward helpers need is present, no matter what the forward saved.
    # TODO: reverse the FFN branch then the attention branch, summing residual + sublayer gradients
    x = cache['attn_branch']['x']
    grads = {}
    block_cache = _complete_block_cache(x, block_params)

    d_ln2, ffn_grads = _ffn_sublayer_backward(d_y, block_cache['ffn_branch']['sublayer_cache'], block_params['ffn'])
    d_attn_out, d_gamma_ln2, d_beta_ln2 = layernorm_backward_affine(d_ln2, block_cache['ffn_branch']['ln_cache'])

    d_attn_out = d_attn_out + d_y

    d_ln1, attn_grads = _attn_sublayer_backward(d_attn_out, block_cache['attn_branch']['sublayer_cache'], block_params['attn'])
    d_x, d_gamma_ln1, d_beta_ln1 = layernorm_backward_affine(d_ln1, block_cache['attn_branch']['ln_cache'])

    grads['ln1'] = {'gamma' : d_gamma_ln1, 'beta' : d_beta_ln1}
    grads['ln2'] = {'gamma' : d_gamma_ln2, 'beta' : d_beta_ln2}
    grads['attn'] = attn_grads
    grads['ffn'] = ffn_grads

    d_x = d_x + d_attn_out
    print(d_x[0, 1, 2])

    return d_x, grads

# Step 140 - stack_transformer_blocks
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    # TODO: create n_layers initialized block parameter dicts and return them as a list
    params_stack = []

    for _ in range(n_layers):
        params = {}
        gamma1 = np.ones(d_model)
        beta1 = np.zeros(d_model)

        attn = create_multihead_qkv_projections(d_model)
        wo = create_multihead_output_projection(d_model)
        bo = np.zeros(d_model)
        attn['Wo']=wo
        attn['bo']=bo

        gamma2 = np.ones(d_model)
        beta2 = np.zeros(d_model)

        w1 = scale_w_small(make_2d_random(d_model, d_ff, 0), 0.02)
        w2 = scale_w_small(make_2d_random(d_ff, d_model, 0), 0.02)
        b1 = np.zeros(d_ff)
        b2 = np.zeros(d_model)

        params['ln1'] = dict(gamma=gamma1,beta=beta1)
        params['attn'] = attn
        params['ln2'] = dict(gamma=gamma2,beta=beta2)
        params['ffn'] = dict(W1=w1,b1=b1,W2=w2,b2=b2)

        params_stack.append(params)
    return params_stack

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    # TODO: thread x through each block in `blocks`, collecting per-block caches
    caches = []
    for params in blocks:
        out = transformer_block_forward(x, params)
        x = out['y']
        caches.append(out['cache'])
    return x, caches

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments (not yet solved)
# TODO: implement

# Step 148 - initialize_adam_step_counter (not yet solved)
# TODO: implement

# Step 149 - adam_increment_step (not yet solved)
# TODO: implement

# Step 150 - adam_update_first_moment (not yet solved)
# TODO: implement

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

