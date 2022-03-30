import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from utils.horovod_model import HorovodModel


class MyModel(HorovodModel):
    
    # 定义模型结构
    def __init__(self):
        super().__init__()
        # model layers, for example
        self.encoder = tf.keras.Sequential([
                tf.keras.Input(shape = (self.input_size, self.input_size, 3)),
                layers.Conv2D(32, 2, strides = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(64, 2, strides = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, 2, strides = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(64, 2, strides = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(32, 2, strides = 2, padding = "same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((self.latent_dim, )),
            ],
            name = "encoder",
        )

    # 模型推理
    def __call__(self, inputs):
        # mdoel forward, return outputs
        return self.encoder(inputs)
    
    # 重写训练步，模型推理计算loss后，调用HorovodModel的train_step更新模型参数
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.build_loss(inputs, outputs)
        return super().train_step(tape = tape, loss = loss)
    
    # 损失函数
    def build_loss(self, inputs, outputs):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(inputs, outputs)
        