''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import  Constants
from Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, pretrained_embeddings=None):

        super().__init__()

        n_position = len_max_seq + 1

        if pretrained_embeddings is None:
            self.src_word_emb = nn.Embedding(
                n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        else:
            self.src_word_emb = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=Constants.PAD, freeze=True)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.segment_enc = nn.Embedding(int(n_position/2), d_word_vec, padding_idx=0)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, src_seg, return_attns=False):
        """
        First creates an imput embedding from the seq, pos, and seg encodings.
        then runs the encoder layer for n_layers and returns the final vector
        
        Args:
            h_seq: Encodings for the words in the history
            h_pos: Positional encodings for the words in the history
            h_seg: Segment encodings for turns in the history
        Returns:
            enc_output: vector output from encoder
        """
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Get input embeddings
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos) \
            + self.segment_enc(src_seg)

        # Nx encoder layer
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, pretrained_embeddings=None):

        super().__init__()
        n_position = len_max_seq + 1

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_seq, enc_output, return_attns=False):
        """
        Starts by getting the imput embedding from the target seq, and pos
        encodings. Then runs the decoder.
        
        Args:
            tgt_seq: Encodings for the words in the target response
            tgt_pos: Positional encodings for the words in the target response
            src_seq: Encodings for the words in the history
            enc_output: Output from the Encoder 
        Returns:
            sec_output: vector outputs from decoder, one for each word in the response 
            
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        #non_pad_mask = get_non_pad_mask(tgt_seq)

        #slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        #slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        #slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        #dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        non_pad_mask = None
        slf_attn_mask = None
        dec_enc_attn_mask = None

        # -- Forward
        dec_output = tgt_seq

        # Nx decoder layer
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_labels, len_max_seq_enc,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            pretrained_embeddings=None):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq_enc,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, pretrained_embeddings=pretrained_embeddings)

        self.tgt_label_prj = nn.Linear(d_model, n_labels, bias=False)
        self.tgt_label_prj = nn.Sequential(nn.Linear(d_model, d_model+512),
                                           nn.ReLU(),
                                           nn.Linear(d_model+512, d_model),
                                           nn.ReLU(),
                                           nn.Linear(d_model, 512),
                                           nn.ReLU(),
                                           nn.Linear(512, n_labels))

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'


    def forward(self, src_seq, src_pos, src_seg, tgt_seq):
        """
        Takes in the input features for the history and response, and returns a prediction.
        
        First encodes the history, and then decodes it before mapping the output to the vocabulary
        
        Args:
            src_seq: Encodings for the words in the history 
            src_pos: Positional encodings for the words in the history 
            src_seg: Segment encodings for turns in the history 
            tgt_seq: Encodings for the words in the target response 
            tgt_pos: Positional encodings for the words in the target response 
        Returns:
            Outputs: Vector of probabilities for each word in the vocabulary, for each word in the response 
        """
        
        tgt_seq = tgt_seq[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos, src_seg)

        outputs = self.tgt_label_prj(enc_output)


        return outputs
