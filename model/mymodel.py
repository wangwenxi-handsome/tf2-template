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
        self.build_metrics()

    # 模型推理
    def __call__(self, inputs):
        # mdoel forward, return outputs
        return self.encoder(inputs)
    
    # 损失函数
    def build_loss(self, labels, preds):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, preds)
    
    # 评价指标
    def build_metrics(self, ):
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        
    # 定义在这里的metrics会在每个epoch开始时重置, 以此保证这些指标是从每个epoch开始计算的
    @property
    def metrics(self):
        return [self.mae_metric]
    
    # 重写训练步，模型推理计算loss后，调用HorovodModel的train_step更新模型参数
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self(inputs)
            loss = self.build_loss(inputs, outputs)
        return super().train_step(tape = tape, loss = loss)
    
    # 重写评估步
    def test_step(self, inputs):
        x, y = inputs
        pred = self(x, training = False)
        loss = self.build_loss(y, pred)
        metrics_dict = {"loss": loss}
        for m in self.metrics:
            m.update_state(y, pred)
            metrics_dict[m.name] = m.result()
        return metrics_dict
    
    # 重写预测步
    def predict_step(self, inputs):
        x = inputs
        preds = self(x, training = False)
        return preds
        
