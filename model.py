# import the modules
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Reshape, Input, concatenate, Layer, Activation, Flatten,
    Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from skimage.metrics import structural_similarity as ssim
import numpy as np
from sklearn.metrics import average_precision_score

from tensorflow.keras.layers import LayerNormalization





class VisionTemporalTransformer(Layer):
    def __init__(self, patch_size, num_patches, d_model, num_heads, num_transformer_layers, temporal_heads, temporal_layers, **kwargs):
        super(VisionTemporalTransformer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.temporal_heads = temporal_heads
        self.temporal_layers = temporal_layers


        # Positional Embedding for spatial patches
        self.position_embedding = tf.Variable(
            tf.random.normal([1, num_patches, d_model]), trainable=True, name="position_embedding"
        )
        self.dense_projection = Dense(d_model)

        # Spatial Transformer Encoder
        self.spatial_encoder_layers = [
            [
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
                LayerNormalization(),
                Dense(d_model * 4, activation="relu"),
                Dense(d_model),
                LayerNormalization(),
            ]
            for _ in range(num_transformer_layers)
        ]


        # Temporal Transformer Encoder
        self.temporal_encoder_layers = [
            [
                tf.keras.layers.MultiHeadAttention(num_heads=temporal_heads, key_dim=d_model),
                LayerNormalization(),
                Dense(d_model * 4, activation="relu"),
                Dense(d_model),
                LayerNormalization(),
            ]
            for _ in range(temporal_layers)
        ]


    def call(self, inputs):
        batch_size, frames, height, width, channels = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            inputs.shape[2],
            inputs.shape[3],
            inputs.shape[4],
        )

        # Flatten frames into (batch_size * frames, height, width, channels)
        reshaped_input = tf.reshape(inputs, [-1, height, width, channels])

        # Extract patches
        patches = tf.image.extract_patches(
            images=reshaped_input,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Flatten patches into (batch_size * frames, num_patches, patch_dim)
        patch_dim = self.patch_size * self.patch_size * channels
        patches = tf.reshape(patches, [-1, self.num_patches, patch_dim])

        # Linear projection and adding positional embeddings
        x = self.dense_projection(patches) + self.position_embedding

        # Spatial Transformer Encoder
        for mha, norm1, ff1, ff2, norm2 in self.spatial_encoder_layers:
            attention_output = mha(x, x)
            x = norm1(x + attention_output)
            ff_output = ff1(x)
            ff_output = ff2(ff_output)
            x = norm2(x + ff_output)

        
        x = tf.reshape(x, [batch_size, frames, self.num_patches, self.d_model])

       
        x = tf.reduce_mean(x, axis=2)  # Shape: (batch_size, frames, d_model)

        # Temporal Transformer Encoder
        for mha, norm1, ff1, ff2, norm2 in self.temporal_encoder_layers:
            attention_output = mha(x, x)
            x = norm1(x + attention_output)
            ff_output = ff1(x)
            ff_output = ff2(ff_output)
            x = norm2(x + ff_output)

       
        pooled_output = GlobalAveragePooling1D()(x)

        return pooled_output


def TemporalTransformer(input_tensor, name_prefix):
    patch_size = 8
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    num_patches = (height // patch_size) * (width // patch_size)
    d_model = 128
    num_heads = 4
    num_transformer_layers = 2
    temporal_heads = 4
    temporal_layers = 2

    transformer = VisionTemporalTransformer(
        patch_size, num_patches, d_model, num_heads, num_transformer_layers, temporal_heads, temporal_layers
    )
    transformer_output = transformer(input_tensor)
    return tf.keras.layers.Reshape((8, 8, 3))(Dense(8 * 8 * 3, activation="relu")(transformer_output))



class MultiHeadAttention(Layer):
    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

    def call(self, queries, keys, values, d_k):
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))
        weights = tf.nn.softmax(scores)
        attention_output = tf.matmul(weights, values)
        return attention_output


def LIPINC_model():
    #Input layer
    frame_input = Input(shape=(8, 64, 144, 3), name="FrameInput")
    residue_input = Input(shape=(7, 64, 144, 3), name="ResidueInput")
    
    # MSTIE
    transformer_frame = TemporalTransformer(frame_input, "Frame")
    transformer_residue = TemporalTransformer(residue_input, "Res")

    
    attention1 = MultiHeadAttention(num_heads=4)
    attention2 = MultiHeadAttention(num_heads=4)
    attention3 = MultiHeadAttention(num_heads=4)
    
    #RGB Mouth Frame branch
    d_k = 192
    color_output = attention1(transformer_residue, transformer_frame, transformer_frame, d_k=d_k)
    color_concat = concatenate([transformer_frame, color_output])
	
    #Delta Frame branch
    structure_output = attention2(transformer_frame, transformer_residue, transformer_residue, d_k=d_k)
    structure_concat = concatenate([transformer_residue, structure_output])
	
    #Fusion
    fusion_output = attention3(structure_concat, color_concat, color_concat, d_k=d_k)

    concatenated = concatenate([structure_concat, fusion_output])

    #MLP	
    conv = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(concatenated)
    conv = BatchNormalization()(conv)
    conv = Conv2D(64, kernel_size=(1, 1), activation="relu", padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(128, kernel_size=(1, 1), activation="relu", padding="same")(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(4, 4))(conv)

    conv = Flatten()(conv)
    conv = Dense(1024, activation="relu")(conv)
    conv = Dense(1024, activation="relu")(conv)
    output = Dense(2, activation="softmax")(conv)

    model = Model(inputs=[frame_input, residue_input], outputs=output)

    return model


def similarity(a, b):
    
    score, _ = ssim(a, b, full=True, multichannel=True)
    return score

#Loss model for inconsistency loss
def total_loss(model):
    def loss(y_true, y_pred):
        frame_features = model.get_layer(index=1).output
        tot = 0
        for i in range(tf.shape(frame_features)[0]):
            for j in range(tf.shape(frame_features)[0]):
                tot += similarity(frame_features[i].numpy(), frame_features[j].numpy())

        avg_sim = tot / (tf.shape(frame_features)[0] * tf.shape(frame_features)[0])
        BCE = BinaryCrossentropy(from_logits=False)
        consistency_loss = BCE(y_true, avg_sim)
        return consistency_loss

    return loss


def LIPINC_V2():
    model = LIPINC_model()
    custom_loss = total_loss(model)
    
    # Optimizer
    opt = Adam(learning_rate=0.001, epsilon=0.1)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", custom_loss], loss_weights=[1, 5],
                  metrics=["accuracy"], run_eagerly=True)
    return model



