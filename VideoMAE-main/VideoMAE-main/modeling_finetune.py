import tensorflow as tf
from tensorflow import keras 
from keras import layers
import numpy as np


def drop_path(x, drop_prob, training):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (tf.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
    random_tensor = tf.floor(random_tensor)
    return x / keep_prob * random_tensor


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

    def get_config(self):
        config = super(DropPath, self).get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = layers.Dense(all_head_dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=None, act_layer='gelu', norm_layer=layers.LayerNormalization, attn_head_dim=None):
        super(Block, self).__init__()
        self.norm1 = norm_layer()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if init_values > 0:
            self.gamma_1 = self.add_weight(shape=(dim,), initializer=tf.keras.initializers.Constant(init_values), trainable=True)
            self.gamma_2 = self.add_weight(shape=(dim,), initializer=tf.keras.initializers.Constant(init_values), trainable=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def call(self, x, training=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)), training=training)
        return x


class PatchEmbed(layers.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.proj = layers.Conv3D(filters=embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), strides=(tubelet_size, patch_size, patch_size))

    def call(self, x):
        B, C, T, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4]
        x = self.proj(x)
        x = tf.reshape(x, (B, -1, tf.shape(x)[1]))
        x = tf.transpose(x, perm=[0, 2, 1])
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return tf.constant(sinusoid_table, dtype=tf.float32)


class VisionTransformer(tf.keras.Model):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=layers.LayerNormalization, init_values=0., use_learnable_pos_emb=False, init_scale=0., all_frames=16, tubelet_size=2, use_checkpoint=False, use_mean_pooling=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight(shape=(1, num_patches, embed_dim), initializer=tf.keras.initializers.Zeros(), trainable=True)
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = layers.Dropout(rate=drop_rate)
        dpr = np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)]
        self.norm = norm_layer() if not use_mean_pooling else layers.LayerNormalization(epsilon=1e-6)
        self.fc_norm = norm_layer() if use_mean_pooling else None
        self.fc_dropout = layers.Dropout(rate=drop_rate)
        self.head = layers.Dense(num_classes, input_dim=embed_dim) if num_classes > 0 else tf.identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed.shape != tf.shape(x):
            self.pos_embed = get_sinusoid_encoding_table(tf.shape(x)[1], self.embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm:
            x = tf.reduce_mean(x, axis=1)
            x = self.fc_norm(x)
        return x

    def call(self, x, training=False):
        x = self.forward_features(x)
        x = self.fc_dropout(x, training=training)
        x = self.head(x)
        return x
