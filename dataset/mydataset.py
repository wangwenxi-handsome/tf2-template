import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers


def preprocess(one_item, *args, **kwargs):
    # 预处理
    # ...
    # for example: jpeg pic
    # pic = tf.io.read_file(one_item)
    # pic = tf.image.decode_jpeg(pic)
    # return pic / 255
    return process_one_item


def build_dataset(file, batch_size, shuffle = False, *args, **kwargs):
    # 从原始数据中构建dataset
    dataset = tf.data.Dataset.from_tensor_slices(file)
    
    # 预处理
    # tf.data.experimental.AUTOTUNE代表tensorflow自动选择合适的参数
    dataset = dataset.map(preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    # 设置训练参数
    # shuffle应在batch前面, buffer_size不宜过大，可能会导致内存混乱，buffer_size越大则越混乱
    # 参数解释
    # 假设数据集的大小为10000，buffer_size的大小为1000，最开始算法会把前1000个数据放入缓冲区；
    # 当从缓冲区的这1000个元素中随机选出第一个元素后，这个元素的位置会被数据集的第1001个元素替换；
    # 然后再从这1000个元素中随机选取第二个元素，第二个元素的位置又会被数据集中的第1002个数据替换，以此类推…
    if shuffle:
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    dataset = dataset.batch(batch_size)
    # 预先准备几个batch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
