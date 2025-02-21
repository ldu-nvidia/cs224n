import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    self.max_length = 1000
    
    # correct way to register causal mask by using upper triangle matrix
    self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.max_length, self.max_length)).view(
                1, 1, self.max_length, self.max_length),)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj
    
  def attention(self, key, query, value, attention_mask):
    # Compute attention scores, normalized by the dimension of of each head
    attention_score = (query @ key.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32)))
    ## use mask fill to fill in inf instead of using arithmetic which will result in NaN
    attention_score = attention_score.masked_fill(self.mask[:, :, :query.shape[-2], :query.shape[-2]] == 0, float('-inf'))
    # Apply softmax to compute attention weights
    attention_weight = self.dropout(torch.softmax(attention_score, dim=-1))
    # Compute final attention output
    attn_value = attention_weight @ value
    # combine multihead into one head
    attn_value = rearrange(attn_value, 'b t h d -> b h (t d)')
    return attn_value

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    
    return attn_value
 