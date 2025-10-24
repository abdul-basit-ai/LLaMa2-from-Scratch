import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    n_kv_heads:Optional[int]=None
    vocab_size : int = -1
    multiple_of:int = 256
    ffn_dim_multiplier : Optional [float]=None
    normm_eps : float = 1e-5
    
    max_batch_size: int = 32
    max_seq_len : int = 2048
    
    device : str = None

def precompute_theta_pos_frequencies(head_dim : int,  seq_len : int , device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0 , "Dimension should be even"
    theta_numerator = torch.arrange(0, head_dim,2).float()
    theta = 1 / (theta ** (theta_numerator/head_dim)).to(device)
    #M
    m = torch.arrange(seq_len,device)
    #Muliply each theta with all posiion using outer product
    freqs = torch.outer(m,theta).float()
    #ComplexForm in polar R * exp(i * theta *m) , r ==1
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs) # turns int0 ei0 equal to represent cos and sin
    
    
    return freqs_complex
    
def apply_rotary_embeddings( x:torch.Tensor, freqs_complex : torch.Tensor, device : str):
    # Batch, seq, h, head_dim/2
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2) )
    #seq_len, head_dim = 1,seq_len,1,head_dim/2 
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #element multiplication results in rotation
    x_rotated = x_complex * freqs_complex
    #transfrom to tensor first real and tgen imaginary
    x_out = torch.view_as_real(x_rotated)
    # b,seq_lem,h,head_dim/2,2 --> B, seq_lem, h , head dim
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
    
class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps : float = 1e-6):
        
        super().__init__()
        self.eps =  eps # to avoid division by 0
        #GAmma Leranable
        self.weight = nn.Parameter(torch.ones(dim))
        
    def norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps) # 1/sqrtt = rsqrt
    
    def forward( self, x:torch.Tensor):
        return self.weight * self.norm(x.float()).type_as(x)

def  repeat_kv(x:torch.Tensor, n_reps : int)-> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_reps == 1:
        return x
    else :
        return(
            x[:,:,:,None,:].expand(batch_size,seq_len,n_kv_heads,n_reps,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_reps,head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q// self.n_kv_heads # number of times kv heeads should be repeated to match q_head
        self.head_dim = args.dim// self.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear (args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear (args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads,self.head_dim))
        self.cache_v =  torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads,self.head_dim))
        
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:int):
        batch_size, seq_len = x.shape
        xq= self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size,seq_len, self.n_heads_q,self.head_dim)
        xk = xk.view(batch_size,seq_len, self.n_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len, self.n_kv_heads,self.head_dim)
        
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv
        
        #reterieve all the caches
        keys = self.cache_k[batch_size,0:start_pos+seq_len]
        values = self.cache_v[batch_size,0:start_pos+seq_len]
        
        #repeat the haeds of kc to match heads of q
        keys = repeat_kv(keys,self.n_reps)
        values = repeat_kv(values,self.n_rep)
        
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
        score = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1,2).contiguous().view(batch_size,seq_len,-1)
        return self.wo(output) 
        
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = 2 *(hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int (hidden_dim * args.ffn_dim_multiplier)
        # Round up to the next multiple of parameter
        hidden_dim= args.multiple_of *((hidden_dim * args.multiple_of -1)// args.multiple_of)
        
        self.w1 = nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,args.dim,bias=False)
        self.w3 = nn.Linear(args.dim,hidden_dim,bias=False)

    def foeward(self,x:torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V= self.w3(x)
        x = swish * x_V
        self.w2(x)
        
        return x
        
class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim//args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, eps=args.normm_eps)
        self.ffn_norm = RMSNorm(args.dim , eps=args.normm_eps)
        
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out    
    
class Transformer(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        
        assert args.vocab_size != -1 , "Set Vocan Size"
        
        self.args = args
        self.n_heads = args.n_heads
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.normm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, Bias = False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos : int):
        # B, SEQ LEN
        batch_size, seq_len = tokens.shape
        assert seq_len ==1 , "Only one token cuz KV cache"
        
        h = self.tok_embeddings(tokens) # into embedding
        
        #get m , theta (start pos, start pos _ seq len)
        freqs_complex = self.freqs_complex[start_pos, start_pos + seq_len]
        for layer in self.layers:
            h = layer(h, start_pos,freqs_complex)
        
        h = self.norm(h)
        output = output(h).float()
        return output