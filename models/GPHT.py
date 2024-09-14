import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GPHTBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()

        self.multipiler = configs.GT_pooling_rate[depth]
        self.patch_size = configs.token_len // self.multipiler
        self.down_sample = nn.MaxPool1d(self.multipiler)
        d_model = configs.GT_d_model
        d_ff = configs.GT_d_ff
        self.patch_embedding = PatchEmbedding(d_model, self.patch_size, self.patch_size, 0, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model,
                        configs.n_heads),
                    d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.GT_e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.forecast_head = nn.Linear(d_model, configs.pred_len)

    def forward(self, x_enc):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # (B, 1, 336)
        # u: [bs * nvars x patch_num x d_model]
        x_enc = self.down_sample(x_enc)  # (B, 1, 42), (B, 1, 84), (B, 1, 168), (B, 1, 336)
        print('down', x_enc.shape)
        enc_out, n_vars = self.patch_embedding.encode_patch(x_enc)
        print('patch', enc_out.shape)
        enc_out = self.patch_embedding.pos_and_dropout(enc_out)
        print('patch_enc_out', enc_out.shape)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        print('enc_out', enc_out.shape)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        print('enc_out2', enc_out.shape)

        bs = enc_out.shape[0]

        # Decoder
        dec_out = self.forecast_head(enc_out).reshape(bs, n_vars, -1)  # z: [bs x nvars x seq_len]
        print('dec_out', dec_out.shape)
        return dec_out.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.encoders = nn.ModuleList([
            GPHTBlock(configs, i)
            for i in range(configs.depth)])

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seq_len = x_enc.shape[1]
        dec_out = 0

        for i, enc in enumerate(self.encoders):
            # print(x_enc.shape)
            out_enc = enc(x_enc)
            dec_out += out_enc[:, -seq_len:, :]
            print(dec_out.shape)
            ar_roll = torch.zeros((x_enc.shape[0], self.configs.token_len, x_enc.shape[2])).to(x_enc.device)
            ar_roll = torch.cat([ar_roll, out_enc], dim=1)[:, :-self.configs.token_len, :]
            x_enc = x_enc - ar_roll

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
