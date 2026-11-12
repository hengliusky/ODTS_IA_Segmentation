import torch.nn as nn
import torch

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, fea_size):
        super(LearnedPositionalEncoding, self).__init__()
        flatten_length = fea_size*fea_size*fea_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, flatten_length, embedding_dim)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings



class Transformer_bottleneck(nn.Module):
    def __init__(self, embedding_dim, fea_dim, out_dim, fea_size, patch_size=1):  # embedding=512
        super(Transformer_bottleneck, self).__init__()
        self.embedding_dim = embedding_dim
        self.fea_size = int(fea_size/patch_size)
        self.fea_dim = fea_dim
        self.out_dim = out_dim

        self.patch_embedding = nn.Conv3d(
            self.fea_dim,
            self.embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.bn = nn.BatchNorm3d(self.fea_dim)
        self.relu = nn.ReLU(inplace=True)

        self.position_encoding = LearnedPositionalEncoding(self.embedding_dim, self.fea_size)

        flatten_length = self.fea_size * self.fea_size * self.fea_size
        self.transformer = TransformerModel(dim=self.embedding_dim, depth=4, heads=8, mlp_dim=flatten_length)

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.conv_xx = nn.Conv3d(
            self.embedding_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            self.fea_size,
            self.fea_size,
            self.fea_size,
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        # patch embedding：通道维 -> embedding_dim，同时完成下采样
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)

        x, _ = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self._reshape_output(x)
        x = self.conv_xx(x)
        return x


if __name__ == '__main__':
    transformer_module = Transformer_bottleneck(
        embedding_dim=512, fea_dim=32, out_dim=32, fea_size=128, patch_size=8
    ).cuda()
    x = torch.rand((1, 32, 128, 128, 128)).cuda()
    y = transformer_module(x)
    print(y.shape)