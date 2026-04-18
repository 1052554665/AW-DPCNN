import torch
import torch.nn as nn

class ConvStem(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)  # [B, D, H/4, W/4]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=drop,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CEViT(nn.Module):
    def __init__(
        self,
        num_classes=10,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.1
    ):
        super().__init__()
        self.in_chans = in_chans

        # NEW: Conv Stem
        self.conv_stem = ConvStem(in_chans, embed_dim)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=8,
            embed_dim=embed_dim
        )

        num_patches = (224 // 4 // 8) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_feat=False):
        B = x.size(0)

        # Accept common grayscale/RGB mismatches without crashing experiments.
        if x.size(1) != self.in_chans:
            if self.in_chans == 1 and x.size(1) == 3:
                x = x.mean(dim=1, keepdim=True)
            elif self.in_chans == 3 and x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise RuntimeError(
                    f"CEViT expected {self.in_chans} input channels, got {x.size(1)}."
                )

        x = self.conv_stem(x)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        feat = x[:, 0]
        logits = self.classifier(feat)

        if return_feat:
            return logits, feat
        else:
            return logits

