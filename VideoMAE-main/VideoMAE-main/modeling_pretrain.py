import tensorflow as tf
from tensorflow import keras 
from keras import layers, models, initializers
import math
import numpy as np

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return tf.convert_to_tensor(sinusoid_table, dtype=tf.float32)

def trunc_normal_(tensor, mean=0.0, std=1.0):
    tensor = initializers.TruncatedNormal(mean=mean, stddev=std).initialize(tensor.shape)
    return tensor

class PretrainVisionTransformerEncoder(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=layers.LayerNormalization, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super(PretrainVisionTransformerEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight("pos_embed", shape=(1, num_patches + 1, embed_dim), initializer='zeros')
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                             init_values=init_values) for _ in range(depth)]
        self.norm = norm_layer(embed_dim)
        self.head = layers.Dense(num_classes, input_dim=embed_dim) if num_classes > 0 else tf.identity()

        if use_learnable_pos_emb:
            self.pos_embed = trunc_normal_(self.pos_embed, std=.02)

    def call(self, x, mask=None):
        x = self.patch_embed(x)

        if mask is not None:
            x = x + tf.cast(self.pos_embed, x.dtype)

            B, _, C = x.shape
            x_vis = tf.reshape(x[~mask], (B, -1, C))

            for blk in self.blocks:
                x_vis = blk(x_vis)

            x_vis = self.norm(x_vis)
            return x_vis
        else:
            x = x + tf.cast(self.pos_embed, x.dtype)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

class PretrainVisionTransformerDecoder(tf.keras.Model):
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=layers.LayerNormalization, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False):
        super(PretrainVisionTransformerDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                             init_values=init_values) for _ in range(depth)]
        self.norm = norm_layer(embed_dim)
        self.head = layers.Dense(num_classes, input_dim=embed_dim) if num_classes > 0 else tf.identity()

    def call(self, x, return_token_num=0):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=768,
                 encoder_depth=12, encoder_num_heads=12, decoder_num_classes=1536, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=layers.LayerNormalization, init_values=0., use_learnable_pos_emb=False,
                 use_checkpoint=False, tubelet_size=2):
        super(PretrainVisionTransformer, self).__init__()
        self.encoder = PretrainVisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans,
                                                        num_classes=encoder_num_classes, embed_dim=encoder_embed_dim, depth=encoder_depth,
                                                        num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                        norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
                                                        use_checkpoint=use_checkpoint, use_learnable_pos_emb=use_learnable_pos_emb)
        self.decoder = PretrainVisionTransformerDecoder(patch_size=patch_size, num_classes=decoder_num_classes, embed_dim=decoder_embed_dim,
                                                        depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                                        norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_checkpoint=use_checkpoint,
                                                        num_patches=self.encoder.patch_embed.num_patches)
        self.encoder_to_decoder = layers.Dense(decoder_embed_dim, use_bias=False)
        self.mask_token = self.add_weight("mask_token", shape=(1, 1, decoder_embed_dim), initializer='zeros')
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        self.mask_token = trunc_normal_(self.mask_token, std=.02)

    def call(self, x, mask):
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = x_vis.shape

        expand_pos_embed = tf.cast(self.pos_embed, x.dtype)
        pos_emd_vis = tf.reshape(expand_pos_embed[~mask], (B, -1, C))
        pos_emd_mask = tf.reshape(expand_pos_embed[mask], (B, -1, C))
        x_full = tf.concat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], axis=1)

        x = self.decoder(x_full, pos_emd_mask.shape[1])
        return x
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, tubelet_size):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = layers.Conv2D(embed_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size))

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [x.shape[0], -1, x.shape[-1]])
        return x

class Block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=layers.LayerNormalization, init_values=None):
        super(Block, self).__init__()
        self.norm1 = norm_layer(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=attn_drop)
        self.drop_path = layers.Dropout(drop_path)
        self.norm2 = norm_layer(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(dim * mlp_ratio, activation='gelu'),
            layers.Dropout(drop),
            layers.Dense(dim),
            layers.Dropout(drop)
        ])

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer= partial(tf.keras.layers.LayerNormalization, epsilon=1e-6),
        **kwargs)
    return model

def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), 
        **kwargs)
    return model

def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer= partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), 
        **kwargs)
    return model

def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(tf.keras.layers.LayerNormalization, epsilon=1e-6), 
        **kwargs)
    return model