import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """
    Triển khai mạng Position-wise Feed-Forward Network.
    Bao gồm hai phép biến đổi tuyến tính với hàm kích hoạt ReLU ở giữa.
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        # Mở rộng chiều dữ liệu lên d_ff (thường là 2048)
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Thu hẹp chiều dữ liệu về lại d_model (thường là 512)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Công thức: FFN(x) = max(0, xW1 + b1)W2 + b2
        """
        # Đưa qua lớp tuyến tính 1, áp dụng ReLU và Dropout
        x = self.dropout(self.relu(self.linear1(x)))
        # Đưa qua lớp tuyến tính 2
        x = self.linear2(x)
        return x

class LayerNorm(nn.Module):
    """
    Lớp chuẩn hóa Layer Normalization.
    Giúp dữ liệu không bị khuếch đại quá lớn hay thu nhỏ quá mức khi đi qua nhiều lớp mạng,
    bằng cách đưa phân phối dữ liệu về trung bình 0 và độ lệch chuẩn 1.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # a_2 (gamma): Tham số scale (tỉ lệ) có thể học được, khởi tạo là vector toàn 1.
        self.a_2 = nn.Parameter(torch.ones(features))
        
        # b_2 (beta): Tham số shift (dịch chuyển) có thể học được, khởi tạo là vector toàn 0.
        self.b_2 = nn.Parameter(torch.zeros(features))
        
        # eps: Một giá trị siêu nhỏ (epsilon) được cộng vào mẫu số để tránh lỗi chia cho 0.
        self.eps = eps  

    def forward(self, x:torch.Tensor):
        # Tính giá trị trung bình (mean) dọc theo chiều cuối cùng (chiều feature), giữ nguyên số chiều
        mean = x.mean(-1, keepdim=True)
        
        # Tính độ lệch chuẩn (standard deviation) dọc theo chiều cuối cùng
        std = x.std(-1, keepdim=True)
        
        # Công thức chuẩn hóa: (x - mean) / (std + eps)
        # Sau đó nhân với tham số scale (a_2) và cộng với tham số shift (b_2)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module): 
    """
    Lớp kết nối tắt (Residual Connection) kết hợp với Layer Normalization và Dropout.
    Đóng vai trò như "hệ tuần hoàn" giúp luồng đạo hàm không bị triệt tiêu khi mạng quá sâu.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # Khởi tạo module LayerNorm đã định nghĩa ở trên
        self.norm = LayerNorm(size)
        
        # Áp dụng Dropout để chống hiện tượng quá khớp (Overfitting)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x, sublayer):
        """
        Nhận đầu vào x và một sublayer (có thể là Multi-Head Attention hoặc Feed Forward).
        """
        # Bước 1: Chuẩn hóa đầu vào ban đầu bằng self.norm(x)
        # Bước 2: Đưa qua module tính toán sublayer(...)
        # Bước 3: Áp dụng ngẫu nhiên tắt một số nơ-ron bằng self.dropout(...)
        # Bước 4: Cộng trực tiếp với đầu vào ban đầu (x + ...) để tạo Residual Connection
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Triển khai một lớp Encoder hoàn chỉnh.
    Bao gồm: Multi-Head Self-Attention và Position-wise Feed-Forward.
    """
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        """
        Khởi tạo EncoderLayer.
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        # Lớp chuẩn hóa (Layer Normalization) cho 2 sub-layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Đầu vào:
            x: Tensor đầu vào có kích thước (batch_size, seq_len, d_model)
            mask: Tensor dùng để che các vị trí padding (padding mask)
        """
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        # Trong self-attention, Query, Key, Value đều đến từ cùng một x
        attn_output = self.self_attn(query=x, key=x, value=x, mask=mask)
        
        # Residual connection và Layer Normalization: LayerNorm(x + Sublayer(x))
        # Dropout được áp dụng vào output của sub-layer trước khi cộng và chuẩn hóa
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Sub-layer 2: Position-wise Feed-Forward ---
        ff_output = self.feed_forward(x)
        
        # Residual connection và Layer Normalization
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
class Encoder(nn.Module):
    """
    Tập hợp N=6 lớp EncoderLayer thành một khối Encoder hoàn chỉnh.
    """
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)  

    def forward(self, x, mask):
        # Thực hiện qua từng lớp encoder
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Triển khai một lớp Decoder hoàn chỉnh.
    Bao gồm: Masked Self-Attention, Encoder-Decoder Attention và Feed-Forward.
    """
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        """
        Khởi tạo DecoderLayer.
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        # Lớp chuẩn hóa (Layer Normalization) cho 3 sub-layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Đầu vào:
            x: Tensor từ phía target (đang tạo ra)
            memory: Tensor output của khối Encoder (chứa thông tin của câu gốc)
            src_mask: Mask che phần padding của câu gốc
            tgt_mask: Mask che phần padding và che các token tương lai (look-ahead mask)
        """
        # --- Sub-layer 1: Masked Multi-Head Self-Attention ---
        # Masking đảm bảo các dự đoán cho vị trí i chỉ dựa vào các vị trí đã biết nhỏ hơn i
        attn_output = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Sub-layer 2: Encoder-Decoder Attention ---
        # Queries đến từ lớp Decoder trước đó, Keys và Values đến từ output của Encoder (memory)
        # Cho phép mỗi vị trí trong Decoder chú ý đến toàn bộ chuỗi đầu vào
        attn_output2 = self.src_attn(query=x, key=memory, value=memory, mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output2))
        
        # --- Sub-layer 3: Position-wise Feed-Forward ---
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
    
class Decoder(nn.Module):
    """ 
    Tập hợp N=6 lớp DecoderLayer thành một khối Decoder hoàn chỉnh.
    """

    def __init__(self, layer, N=6):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N=6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class Generator(nn.Module):
    '''
    Lớp Generator dùng để chuyển đổi output cuối cùng của Decoder thành xác suất phân phối trên từ vựng.
    '''
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
class EncoderDecoder(nn.Module):
    '''
    Mô hình Transformer hoàn chỉnh bao gồm Encoder, Decoder, Embeddings và Generator.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, 
    generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator  

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Bước 1: Mã hóa câu nguồn
        memory = self.encode(src, src_mask)
        
        # Bước 2: Giải mã
        decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)
        
        # Bước 3: Đẩy qua Generator để biến thành xác suất phân bố từ vựng (Khắc phục lỗi shape ở đây)
        final_output = self.generator(decoder_output)
        
        return final_output

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(
                self.tgt_embed(tgt), memory, src_mask, tgt_mask
            )