# LLaMa2-from-Scratch
### For detailed notes, check out pdf. (Checkout Paligemma repo)


## Key Concepts

### Multi-Query Head Attention (MQHA)

- Reduces memory bandwidth by using multiple query heads but only single key/value heads
- Significantly faster inference with minimal quality impact
- Lower KV cache memory requirements during generation

### RoPE (Rotary Position Embeddings)

- Encodes absolute position information with relative position properties
- Applies rotations to query/key representations in attention mechanism
- Better extrapolation to longer sequences than learned positional embeddings

### RMSNorm (Root Mean Square Normalization)

- Simpler and faster alternative to LayerNorm
- Normalizes using RMS statistics instead of mean and variance
- Improves training stability with lower computational cost

### SwiGLU Activation

- Gated Linear Unit variant using Swish activation
- FFN uses three linear projections: gate, up, and down
- Better performance than standard ReLU-based FFNs

### Grouped-Query Attention (GQA)

- Hybrid between multi-head and multi-query attention
- Multiple query heads share groups of key/value heads
- Balances quality and inference speed (used in 70B model)

<img width="594" height="811" alt="image" src="https://github.com/user-attachments/assets/d3cd1f52-a0da-4fd6-8250-ccdebf76403d" />
