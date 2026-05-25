import jax.numpy as jnp 
import jax
import flax.nnx as nnx


class PositionalEncoding(nnx.Module):
    def __init__(self, head_dim, theta=10000.0):
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, q, k):
        seq_len = q.shape[-2]
        positions = jnp.arange(seq_len)[:, None]
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        angles = positions * inv_freq
        cos = jnp.cos(angles)[None, None, :, :]
        sin = jnp.sin(angles)[None, None, :, :]

        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)

    def _apply_rope(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = jnp.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            axis=-1,
        )
        return x_rotated.reshape(x.shape)


class TransformerLayer(nnx.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dim, 
                 num_heads,
                 dropout,
                 use_rope=True,
                 rope_theta=10000.0,
                 use_bias=True,
                 rng:nnx.Rngs=None,
                 **kwargs):

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope

        self.linear_q = nnx.Linear(in_dim, hidden_dim, use_bias=use_bias, rngs=rng)
        self.linear_k = nnx.Linear(in_dim, hidden_dim, use_bias=use_bias, rngs=rng)
        self.linear_v = nnx.Linear(in_dim, hidden_dim, use_bias=use_bias, rngs=rng)
        self.linear_out = nnx.Linear(hidden_dim, in_dim, use_bias=use_bias, rngs=rng)
        self.positional_encoding = PositionalEncoding(self.head_dim, theta=rope_theta)

        self.attn_layer_norm = nnx.LayerNorm(in_dim, use_bias=use_bias, rngs=rng)
        self.mlp_layer_norm = nnx.LayerNorm(in_dim, use_bias=use_bias, rngs=rng)
        
        # define MLP 
        self.mlp = nnx.Sequential(
            nnx.Linear(in_dim, hidden_dim, use_bias=use_bias, rngs=rng),
            jax.nn.gelu,
            nnx.Linear(hidden_dim, in_dim, use_bias=use_bias, rngs=rng),
        )

        self.dropout = nnx.Dropout(rate=dropout, rngs=rng)


    def __call__(self, x, causal_mask=True, deterministic=False): 
        batch_size, seq_len, _ = x.shape

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.use_rope:
            q, k = self.positional_encoding(q, k)
        
        # attention: scaled dot-product per head
        attn = (q @ k.transpose(0,1,3,2)) * (1 / jnp.sqrt(self.head_dim))  # (batch, heads, seq, seq)
        if causal_mask:
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            attn = jnp.where(mask, attn, -jnp.inf)
        attn = jax.nn.softmax(attn, axis=-1)
        x_attn = attn @ v  # (batch, heads, seq, head_dim)
        
        # Concatenate heads
        x_attn = x_attn.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        x_attn = x_attn.reshape(x_attn.shape[0], x_attn.shape[1], -1)  # (batch, seq, hidden_dim)
        x_attn = self.linear_out(x_attn)
        x_attn = self.dropout(x_attn, deterministic=deterministic)
        x = self.attn_layer_norm(x + x_attn)  # Add residual connection

        x_mlp = self.mlp(x)
        x_mlp = self.dropout(x_mlp, deterministic=deterministic)
        x = self.mlp_layer_norm(x + x_mlp)  # Add residual connection

        return x


class Transformer(nnx.Module):
    def __init__(self,
                 vocab_size,
                 model_dim,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 dropout,
                 use_bias=True,
                 use_rope=True,
                 rng:nnx.Rngs=None,
                 **kwargs):
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers

        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=model_dim,
            rngs=rng,
        )
        self.layers = nnx.List([
            TransformerLayer(
                in_dim=model_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=use_rope,
                use_bias=use_bias,
                rng=rng,
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nnx.LayerNorm(model_dim, use_bias=use_bias, rngs=rng)
        self.output = nnx.Linear(model_dim, vocab_size, use_bias=False, rngs=rng)

    def __call__(self, tokens, deterministic=False):
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x, causal_mask=True, deterministic=deterministic)

        x = self.layer_norm(x)
        return self.output(x)



