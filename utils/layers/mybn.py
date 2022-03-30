import tensorflow as tf
from tensorflow import keras 
import horovod.tensorflow as hvd


# 自定义的分布式BN
# BN层的注意事项
# https://blog.csdn.net/flash_zhj/article/details/107071104
class MyBn(tf.keras.layers.Layer):

    def __init__(self,
                 axis = -1,
                 momentum = 0.99,
                 epsilon = 0.001,
                 bn_freeze_steps = None,
                 trainable = True):
        super(MyBn, self).__init__()
        # 推理时的均值和方差是通过 指数加权移动平均法EWMA 计算训练时每个batch的均值和方差得到的
        # moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
        # moving_var = moving_var * momentum + var(batch) * (1 - momentum)
        # # 若batch只有10组，设置为0.9；若为50，设置为0.98；若大于100，可保持默认值0.99，否则会出现训练和测试时均值和方差差距过大
        self.momentum = momentum
        # 添加到方差分母中的小浮点数，防止被0除
        self.epsilon = epsilon
        # axis一般指向特征轴，只保留这个维度，分别计算每个特征的均值和方差
        self.axis = axis
        # 参数更新的步数
        self.bn_freeze_steps = bn_freeze_steps
        self.trainable = trainable
        self.bn = tf.keras.layers.BatchNormalization(axis=axis,
                                                     momentum=self.momentum,
                                                     epsilon=self.epsilon,
                                                     fused=True)

    def call(self, inputs, training=False):
        if training:
            # 获取步数
            step = tf.summary.experimental.get_step()
            if self.bn_freeze_steps is None or step < self.bn_freeze_steps:
                output = self.bn(inputs, training=True)
                self.bn.moving_mean.assign(
                    hvd.allreduce(self.bn.moving_mean.read_value()))
                self.bn.moving_variance.assign(
                    hvd.allreduce(self.bn.moving_variance.read_value()))
            else:
                output = self.bn(inputs, training=False)
        else:
            output = self.bn(inputs, training=False)

        return output
