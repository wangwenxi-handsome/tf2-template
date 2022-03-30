import tensorflow as tf
import math


# 重写tf.keras.optimizers.schedules.LearningRateSchedule中的__init__和__call__实现自定义的shedule
# call中实现用了一种近似方法，简洁很多
class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, *, lr_fn):
        self.lr_fn = lr_fn

    def __call__(self, step):
        lr = self.lr_fn(step)
        # tf.summary.scalar('learning_rate', lr)
        return lr


class LinearLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, *, base_lr, max_steps, warmup_steps_rate):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = max_steps * warmup_steps_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = (1.0 - step / self.max_steps) * self.base_lr
        warmup_lr = step / self.warmup_steps * self.base_lr
        lr = tf.maximum(tf.minimum(lr, warmup_lr), 0.0)
        # tf.summary.scalar('learning_rate', lr)
        return lr


class CosineLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, *, base_lr, max_steps, warmup_steps_rate):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = max_steps * warmup_steps_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = 0.5 * (1.0 + tf.math.cos(math.pi * step / self.max_steps)) * self.base_lr
        warmup_lr = step / self.warmup_steps * self.base_lr
        lr = tf.maximum(tf.minimum(lr, warmup_lr), 0.0)
        # tf.summary.scalar('learning_rate', lr)
        return lr
