"""
Transformerモジュールどもをscratchで実装
"""
import collections
import torch
import torch.nn as nn
import numpy as np

# sinusoidal position embedding
class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len: int,embed_model_dim: int):
        """
        Args:
            max_seq_len(int) : length of input sequence
            embed_model_dim(int) : dimension of embedding
        """
        super(PositionalEmbedding,self).__init__()
        self.embed_dim = torch.tensor(embed_model_dim).float()
        #self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,embed_model_dim)
        for pos in range(max_seq_len):
            for i in range(0,embed_model_dim,2):
                pe[pos,i] = torch.sin(torch.tensor(pos/(10000**(2*i/embed_model_dim))))
                pe[pos,i+1] = torch.cos(torch.tensor(pos/(10000**(2*i/embed_model_dim))))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        """
        Args:
            x(torch.Tensor) : input tensor (B,Length,Dim)
        Returns:
            torch.Tensor : input tensor + positional embedding
        """
        x = x + torch.sqrt(self.embed_dim) # make embeddings relatively larger
        #x = x + math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)
        return x

# transformer modules 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,dim: int,num_heads :int=8,qkv_bias: bool=True,dropout: float=0.,
                 is_causal: bool=False,quiet_attention: bool=False):
        """
        Args:
            dim (int): 埋め込み次元数
            num_heads (int): MultiHeadAttentionのHead数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            is_causal (bool): Trueの場合、masked multi-head attentionを行う
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
        Note:
            quiet attentionのreference
            https://www.evanmiller.org/attention-is-off-by-one.html
        """
        super().__init__()
        
        self.is_causal = is_causal
        self.quiet_attention = quiet_attention
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"The hidden size {dim} is not a multiple of the number of head attention"
        self.hidden_dim = dim
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim,dim,bias=qkv_bias)
        self.key = nn.Linear(dim,dim,bias=qkv_bias)
        self.value = nn.Linear(dim,dim,bias=qkv_bias)
        
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Dropout(p=dropout),
        )
    
    def forward(self,x,mask=False):
        """
        Args:
            x (torch.Tensor): input tensor (B,Length,Dim)
            mask (bool): Trueの場合、masked multi-head attentionを行う
        """
        batch_size,num_patches,_ = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # マルチヘッドに分割
        #multihead_qkv_shape = q.size()[:-1] + (self.num_heads, self.head_dim)
        multihead_qkv_shape = torch.Size([batch_size, num_patches, self.num_heads, self.head_dim])
        qs = q.view(multihead_qkv_shape)
        qs = qs.permute(0, 2, 1, 3)
        ks = k.view(multihead_qkv_shape)
        ks = ks.permute(0, 2, 1, 3)
        ks_T = ks.transpose(2,3)
        vs = v.view(multihead_qkv_shape)
        vs = vs.permute(0, 2, 1, 3)
        
        scaled_dot_product = qs@ks_T / np.sqrt(self.head_dim)

        # masked multi-head attention
        if self.is_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(num_patches,device=x.device)
            scaled_dot_product = scaled_dot_product + mask

        if self.quiet_attention:
            self_attention = _softmax_one(scaled_dot_product,dim=-1)
        else:
            self_attention = nn.functional.softmax(scaled_dot_product,dim=-1)
        self_attention = self.dropout(self_attention) # 実装上はあるっぽいけど何なんこれ
        
        context_layer = self_attention@vs
        #context_layer = context_layer.transpose(1,2).reshape(batch_size,num_patchs,self.hidden_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().reshape(batch_size,num_patches,self.hidden_dim)
        out = self.projection(context_layer)
        #out = context_layer
        
        return out

def _softmax_one(x,dim=-1):
    """ https://www.evanmiller.org/attention-is-off-by-one.html の実装
    Args:
        x (torch.Tensor):
        dim (int, optional): softmaxを取る次元. Defaults to -1.
    Returns:
        torch.Tensor: softmaxを取った後のテンソル
    """
    x = x - x.max(dim=dim, keepdim=True).values # subtract the max for stability
    exp_x = torch.exp(x)
    return exp_x / (1+exp_x.sum(dim=dim,keepdim=True))

class FeedForward(nn.Module):
    def __init__(self,dim: int,hidden_dim: int=768*4,activation=nn.GELU(),dropout: float=0.):
        """
        Args:
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            dropout (float): ドロップアウト確率
        """
        super().__init__()
        self.linear1 = nn.Linear(dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,dim)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x

# GPT modules
class GPTBlocks(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
        """
        Args:
            embed_dim (int): 埋め込み次元数
            num_heads (int): MultiHeadAttentionのHead数
            ff_hidden_dim (int): FeedForward Networkの隠れ層次元数
            dropout (float): ドロップアウト確率
        """
        super(GPTBlocks,self).__init__()

        self.mmhsa = MultiHeadSelfAttention(dim=embed_dim,num_heads=num_heads,dropout=dropout,is_causal=True)
        self.ff = FeedForward(dim=embed_dim,hidden_dim=ff_hidden_dim,dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        z = self.mmhsa(x)
        z = self.norm1(z)
        x = x + z
        z = self.ff(x)
        z = self.norm2(z)
        x = x + z
        return x


# main models
class GPT(nn.Module):
    def __init__(self, max_seq_len: int, vocab_size: int, num_blocks: int,
                 embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float):
        """
        Args:
            max_seq_len (int): 入力系列の最大長
            vocab_size (int): 語彙数
            embed_dim (int): 埋め込み次元数
            num_blocks (int): TransformerBlockの数
            num_heads (int): MultiHeadAttentionのHead数
            ff_hidden_dim (int): FeedForward Networkの隠れ層次元数
            dropout (float): ドロップアウト確率
        """
        super(GPT,self).__init__()
        self.max_seq_len = max_seq_len

        self.embedding_layer = nn.Embedding(vocab_size,embed_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len,embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            GPTBlocks(embed_dim=embed_dim,num_heads=num_heads,ff_hidden_dim=ff_hidden_dim,dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.head = nn.Linear(embed_dim,vocab_size) # embeddingの逆行列を使う方法もある
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x,targets=None):
        """
        Args:
            x (torch.Tensor): 入力トークン (batch_size,seq_len)
            target (torch.Tensor): 教師トークン (batch_size,seq_len)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: GPTの出力 (batch_size,seq_len,vocab_size), 損失 (1,)
        """
        x = self.embedding_layer(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.softmax(x)

        if targets is not None:
            loss = nn.functional.cross_entropy(x.view(-1,x.size(-1)),targets.view(-1))
            return x,loss
        else:
            return x,None
    
    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.0):
        """
        Args:
            idx (list[int]): 入力トークンのリスト
            max_new_tokens (int): 生成するトークン数
            temperature (float): softmaxの温度
        Returns:
            list[int]: 生成されたトークンのリスト
        TODO:
            ちゃんと実装する
        """
        self.eval()
        for _ in range(max_new_tokens):
            if idx.size(1) < self.max_seq_len:
                idx = idx[:,-self.max_seq_len:]
            x = idx.clone().detach().view(1,-1)
            y,_ = self(x)
            idx_next = y.argmax(-1)[:,-1]
            idx_next = idx_next.unsqueeze(-1).clone().detach()
            idx = torch.cat((idx,idx_next),dim=1)

        return idx
