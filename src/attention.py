import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
 
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Tính Scaled Dot-Product Attention.
 
    Công thức: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 
    Args:
        query : Tensor shape (batch, heads, seq_q, d_k)
        key   : Tensor shape (batch, heads, seq_k, d_k)
        value : Tensor shape (batch, heads, seq_k, d_v)
        mask  : Tensor shape (batch, 1, 1, seq_k) hoặc (batch, 1, seq_q, seq_k)
                  - padding mask  : che các vị trí <PAD>
                  - look-ahead mask: che các token tương lai trong decoder
        dropout: nn.Dropout module (tùy chọn)
 
    Returns:
        output      : Tensor shape (batch, heads, seq_q, d_v)
        attn_weights: Tensor shape (batch, heads, seq_q, seq_k)  — dùng để visualize
    """
    d_k = query.size(-1)
 
    # --- Bước 1: Tính điểm tương đồng (attention scores) ---
    # QK^T / sqrt(d_k)  →  shape: (batch, heads, seq_q, seq_k)
    # Chia cho sqrt(d_k) để tránh gradient bị triệt tiêu khi d_k lớn
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
 
    # --- Bước 2: Áp dụng mask (nếu có) ---
    # Điền -inf vào các vị trí bị che → sau softmax sẽ thành 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
 
    # --- Bước 3: Softmax → attention weights ---
    # Các trọng số dương, tổng = 1 trên chiều seq_k
    attn_weights = F.softmax(scores, dim=-1)
 
    # Xử lý trường hợp toàn bộ một hàng bị mask (tránh NaN từ softmax(-inf,...,-inf))
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
 
    # --- Bước 4: Dropout trên attention weights (theo paper) ---
    if dropout is not None:
        attn_weights = dropout(attn_weights)
 
    # --- Bước 5: Nhân với Value ---
    # (batch, heads, seq_q, seq_k) x (batch, heads, seq_k, d_v)
    # → (batch, heads, seq_q, d_v)
    output = torch.matmul(attn_weights, value)
 
    return output, attn_weights
 
 
class MultiHeadAttention(nn.Module):
    """
    Triển khai Multi-Head Attention.
 
    Thay vì tính attention một lần trên không gian d_model,
    ta chiếu Q/K/V xuống h không gian con d_k = d_model/h song song,
    rồi ghép (concatenate) kết quả lại → phong phú hơn về biểu diễn.
 
    Công thức:
        MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_O
        head_i           = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
 
    Args:
        d_model : Chiều của model (thường là 512)
        h       : Số attention heads (thường là 8)
        dropout : Tỉ lệ dropout áp dụng lên attention weights
    """
 
    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model phải chia hết cho số heads h"
 
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # Chiều của mỗi head
 
        # --- 4 ma trận chiếu tuyến tính (không có bias theo paper gốc) ---
        # W_Q, W_K, W_V: chiếu đầu vào xuống không gian heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
 
        # W_O: chiếu concatenated output về lại d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)
 
        self.dropout = nn.Dropout(p=dropout)
 
        # Lưu attention weights để tiện visualize / debug
        self.attn_weights = None
 
    def forward(self, query: torch.Tensor,
                      key:   torch.Tensor,
                      value: torch.Tensor,
                      mask:  torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query : (batch, seq_q, d_model)
            key   : (batch, seq_k, d_model)
            value : (batch, seq_k, d_model)
            mask  : (batch, 1, seq_q, seq_k) hoặc (batch, 1, 1, seq_k)
 
        Returns:
            output: (batch, seq_q, d_model)
        """
        batch_size = query.size(0)
 
        # =========================================================
        # Bước 1 — Chiếu tuyến tính Q, K, V
        # =========================================================
        # (batch, seq, d_model) → (batch, seq, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
 
        # =========================================================
        # Bước 2 — Tách thành h heads
        # (batch, seq, d_model) → (batch, seq, h, d_k) → (batch, h, seq, d_k)
        # =========================================================
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
 
        # =========================================================
        # Bước 3 — Scaled Dot-Product Attention song song trên h heads
        # output: (batch, h, seq_q, d_k)
        # =========================================================
        attn_output, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
 
        # =========================================================
        # Bước 4 — Ghép các heads lại (Concatenate)
        # (batch, h, seq_q, d_k) → (batch, seq_q, h, d_k) → (batch, seq_q, d_model)
        # =========================================================
        attn_output = (attn_output
                       .transpose(1, 2)                          # (batch, seq_q, h, d_k)
                       .contiguous()
                       .view(batch_size, -1, self.d_model))      # (batch, seq_q, d_model)
 
        # =========================================================
        # Bước 5 — Chiếu output lần cuối qua W_O
        # =========================================================
        return self.W_o(attn_output)
