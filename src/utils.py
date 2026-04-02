import copy
import torch.nn as nn

def clones(module, N):
    """
    Tạo ra N bản sao độc lập (deep copy) của một PyTorch module.
    
    Args:
        module (nn.Module): Lớp mạng cần sao chép (ví dụ: EncoderLayer)
        N (int): Số lượng bản sao cần tạo (ví dụ: 6)
        
    Returns:
        nn.ModuleList: Danh sách chứa N module đã được nhân bản
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])