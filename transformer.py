"import thư viện"

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import math


##################################################################

'''
    class Embedding
    Đối số đầu vào: 
        + voc_size: Kích thước của tập từ vựng
        + d_model: Kích thước của vector từ mong muốn biểu diễn
'''

class Embedding(nn.Module):
    def __init__(self, voc_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(voc_size,d_model)
    def forward(self,x):
        return self.embedding(x)
    

##################################################################

'''
    class PositionEncoding: Lưu thông tin vị trí của mỗi pos trong vector embbeding
    Đối số đầu vào:
        + d_model: chiều dài vector embedding
        + max_length: chiều dài tối đa của câu
'''

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_length=500):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        """ 
            - Duyệt qua toàn bộ câu 
            - Duyệt qua toàn bộ vector
            - Tính toán giá trị của pos
        """
        pe = torch.zeros(max_length,d_model)
        for token_id in range(max_length):
            for index in range(0,d_model,2):
                w_k = 1/(10000**(index/d_model))
                pe[token_id,index] = math.sin(w_k*token_id)
                pe[token_id,index+1] = math.cos(w_k*token_id)
        pe = pe.unsqueeze(0)
        self.weight = Variable(pe, requires_grad=False)
    def forward(self, x):
        if x.is_cuda:
            self.weight = self.weight.to('cuda')
            return  x + self.weight[:,x.size(1),:]
        else:
            return  x + self.weight[:,x.size(1),:]

##################################################################

'''
    Class ScaleDotProduct: Tính giá trị attention của 3 vector K,Q,V
    Đối số đầu vào:
        + drop: Giá trị bỏ học, mặc định .1 

'''

class ScaleDotProduct(nn.Module):
    def __init__(self, drop=.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
    def forward(self, Q, K, V, mask=None):
        d_k = K.size(-1)   # Chiều dài của vector key, chiều dài này khác chiều dài d_model
        attention_scores = torch.matmul(K,Q.transpose(-2,-1))/math.sqrt(d_k)
        attention_scores = F.softmax(attention_scores,-1)
        if mask != None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0,0)
        attention_scores = self.drop(attention_scores)
        return torch.matmul(attention_scores,V)


##################################################################

"""
    Class HeadAttention: thực hiện tìm giá trị 3 vector K, Q, V sau đó tính attention 
    Đối số đầu vào:
        - d_model: chiều dài của vector embedding
        - d_feature: chiều dài của vector K,Q,V sau khi nhân matrix input với từng W_k, W_q và W_v tương ứng
"""

class HeadAttention(nn.Module):
    def __init__(self, d_model=512, d_feature=64, drop=.1):
        super().__init__()
        
        # Khởi tạo 3 ma trận trọng số W_k, W_q, W_q
        self.W_k = nn.Linear(d_model, d_feature)
        self.W_q = nn.Linear(d_model, d_feature)
        self.W_v = nn.Linear(d_model, d_feature)
        self.attention = ScaleDotProduct(drop)

    def forward(self, queries, keys, values, mask=None):
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)
        return self.attention(Q,K,V,mask)

##################################################################
"""
    Class MultiHeadAttention: Thực hiện tính attention trên nhiều head
    Đối số đầu vào:
        + n_heads: Số lượng head
        + d_model: Kích thước vector embedding
        + d_feature: Kích thước vector K
"""

class  MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_feature= 64, drop=.1):
        super().__init__()
        self.n_heads = n_heads
        self.attentions = nn.ModuleList(HeadAttention(d_model=d_model,d_feature=d_feature,drop = drop) for _ in range(n_heads))

        # Tạo một ma trận W_0 để trả về kích thước ban đầu
        self.W_0 = nn.Linear(n_heads*d_feature, d_model)
    
    def forward(self, queries, keys, values, mask=None):
        x = [attention(queries, keys, values, mask) for _, attention in enumerate(self.attentions)]
        x = torch.cat(x, dim=-1)
        return self.W_0(x)

##################################################################
"""
    Class Encoder: Mã hoá một câu đầu vào biểu diễn thành các vector ngữ nghĩa
    Đối số đầu vào:
        + d_model: Kích thước vector nhúng từ
        + d_feature: Kích thước của vector K
        + dff: Số lượng node trong feedforward
        + n_heads: số lượng head
"""

class Encoder(nn.Module):
    def __init__(self, d_model=512, d_feature=64, d_ff=2048, n_heads=8, drop=.1):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, d_feature=d_feature, 
                                                    drop=drop)
        self.norm1_layer = nn.LayerNorm(d_model)
        self.norm2_layer = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(.1),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self,x, mask=None):
        features = self.MultiHeadAttention(x, x, x, mask=mask)

        #norm
        norm1 = self.norm1_layer(features)

        # skip connection
        x = x + self.drop(norm1)

        #feed forward
        ff = self.feed_forward(x)

        # norm
        norm2 = self.norm2_layer(ff)

        # skip connection
        return x + self.drop(norm2)


    
##################################################################
"""
    Class Encoders: Mã hoá một câu đầu vào biểu diễn thành các vector ngữ nghĩa
    Đối số đầu vào:
        + d_model: Kích thước vector nhúng từ
        + d_feature: Kích thước của vector K
        + dff: Số lượng node trong feedforward
        + n_heads: số lượng head
        + n_layers: Số lượng Encoder
"""
class Encoders(nn.Module):
    def __init__(self, d_model=512, d_feature=64, d_ff=2048,
                        n_heads=8, n_layers=10,  drop=.1,max_length=500):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(d_model, d_feature, d_ff,n_heads, drop) for _ in range(n_layers)])
        self.position_encoding = PositionEncoding(d_model,max_length)
    def forward(self, x: torch.FloatTensor, mask = None):
        # Mã hoá vị trí trong câu
        x = self.position_encoding(x).float()
        # Mã hoá
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x

##################################################################
"""
    Class Decoder: Giải mã 
    Đối số đầu vào:
        + d_model: Kích thước vector nhúng từ
        + d_feature: Kích thước của vector K
        + dff: Số lượng node trong feedforward
        + n_heads: số lượng head
"""
class Decoder(nn.Module):
    def __init__(self, d_model=512, d_feature=64, d_ff=2048, n_heads=8, drop=.1):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(n_heads=n_heads, d_feature=d_feature, 
                                                            d_model=d_model, drop=drop)
        self.drop = nn.Dropout(drop)
        self.norm1_layer = nn.LayerNorm(d_model)
        self.norm2_layer = nn.LayerNorm(d_model)
        self.norm3_layer = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.LeakyReLU(.1),
            nn.Linear(d_ff,d_model)
        )
    def forward(self, encoder_output, x, encoder_mask=None , decoder_mask=None):

        # Tính attention mask
        mask_attentions = self.MultiHeadAttention(x, x, x, mask = decoder_mask)

        # norm kq vừa tính
        norm1 = self.norm1_layer(mask_attentions)

        # skip connection
        x = x + self.drop(norm1)

        # Tính attention lần 2
        #key, value lấy từ encoder_output
        attentions = self.MultiHeadAttention(x, encoder_output, encoder_output, mask=encoder_mask)

        # norm
        norm2 = self.norm2_layer(attentions)

        # skip connection
        x = x + self.drop(norm2)

        # ff
        x = self.feed_forward(x)

        # norm
        norm3 = self.norm3_layer(x)

        # skip connection
        return x + self.drop(norm3)
        
##################################################################
"""
    Class Decoders: Giải mã 
    Đối số đầu vào:
        + d_model: Kích thước vector nhúng từ
        + d_feature: Kích thước của vector K
        + dff: Số lượng node trong feedforward
        + n_heads: số lượng head
        + n_layers: số lượng decoder
"""
class Decoders(nn.Module):
    def __init__(self, d_model=512, d_feature=64, d_ff=2048, n_heads=8,
                         n_layers=10, drop=.1, max_length=500):
        super().__init__()
        self.decoders = nn.ModuleList([Decoder(d_model, d_feature, d_ff, n_heads, drop) for _ in range(n_layers)])
        self.position_encoding = PositionEncoding(d_model, max_length)

    def forward(self, output_encoder, x, encoder_mask = None, decoder_mask = None):
        x = self.position_encoding(x)
        for decoder in self.decoders:
            x = decoder(output_encoder, x, encoder_mask , decoder_mask)
        return x


##################################################################
"""
    Class Transformer: Kết hợp Encoders và Decoders
"""

class Transformer(nn.Module):
    def __init__(self, voc_size, d_model=512, d_feature=64, d_ff = 2048,
                    n_heads = 8, n_layers=10, max_length=500, drop=.1):
        super().__init__()
        self.encoders = Encoders(d_model=d_model, d_feature = d_feature, d_ff = d_ff,
                            n_heads=n_heads, n_layers = n_layers, max_length=max_length, drop = drop)
        self.decoders = Decoders(d_model=d_model, d_feature = d_feature, d_ff = d_ff,
                            n_heads=n_heads, n_layers = n_layers, max_length=max_length, drop = drop)
        self.embbeding = Embedding(voc_size, d_model)
        self.fc = nn.Linear(d_model, voc_size)
    
    def forward(self,input_encoder, input_decoder, encoder_mask = None, decoder_mask=None):
        encode_embedding = self.embbeding(input_encoder)
        decode_embedding = self.embbeding(input_decoder)
        output_encoders = self.encoders(encode_embedding,mask = encoder_mask)
        output_decoders = self.decoders(output_encoders,decode_embedding,encoder_mask,decoder_mask)
        return self.fc(output_decoders)


# transformer = Transformer(100)
# en = torch.LongTensor([[1,2,3,4]])
# de = torch.LongTensor([[5,6,7,8]])
# em = Embedding(100,512)(en)

# input = torch.rand(3,5,512)
# keys = torch.rand((3,5,512))
# values = torch.rand((3,5,512))
# queries = torch.rand((3,5,512))
# mask2 = torch.randint(0,2,(1,4))
# print(transformer(en, de, mask1, mask2))
# # print(queries.mask_filled(mask==0,0))
# # print(mask.unsqueeze(1))
# # ScaleDotProduct()(queries,keys,values,mask)

# x = torch.LongTensor(torch.randint(0,2,(2,1000)))
# print(Embedding(1,512)(x).shape)

