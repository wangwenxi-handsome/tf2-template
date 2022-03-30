import os
import random
import tensorflow as tf      
from tensorflow import keras
import numpy as np


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

# 无需源代码，加载后即可使用模型
# https://www.tensorflow.org/guide/saved_model?hl=zh-cn
def save_model(model, logdir):
    tf.saved_model.save(model, logdir)