import os
import argparse
import tensorflow as tf
import horovod.tensorflow as hvd
import configs
from utils.utils import setup_seed
from utils.horovod_model import hvd_init
from utils import layers
from model.mymodel import MyModel
from dataset.mydataset import build_dataset
from utils.utils import save_model


def main(args):
    # 设置随机种子
    setup_seed(configs.train.seed)
    
    # hvd初始化
    hvd_init()
    
    # 设置是否混合精度
    tf.keras.mixed_precision.experimental.set_policy(
        ('float32', 'mixed_float16')[configs.train.mixed_precision])
    
    # 初始化dataset
    # hvd训练时需要提前对数据集进行划分
    dataset = build_dataset(file = f"data_{hvd.rank()}.txt", batch_size = configs.train.batch_size)
    
    # 初始化模型
    model = MyModel()
    
    # 设置优化器
    # 关于自定义学习率衰减 https://blog.csdn.net/Light2077/article/details/106629697
    # 将lr_schedule作为optimizer的lr输入
    lr_schedule = layers.LinearLRSchedule(
        base_lr = configs.train.lr,
        max_steps = configs.train.epochs * len(dataset),
        warmup_steps = configs.train.warm_up_rate * configs.train.epochs * len(dataset),
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule, epsilon = 1e-3)
    
    # compile，传入optimizer即可
    # run_eagerly = True是以动态图的形式运行，类似于python的运行方式，但速度慢
    # run_eagerly = False是编译成静态图的形式运行，速度快，打印中间变量需使用tf.print
    model.compile(optimizer,
                  dynamic_loss_scale=configs.train.mixed_precision,
                  run_eagerly=False)
    
    # 加载checkpoint
    model.restore_or_initialize(logdir = args.logdir)
    
    # train or test
    if args.mode == "train":
        model.fit(dataset, logdir = args.logdir, epochs = configs.train.epochs)
        
    # save model, 只在主进程保存
    if args.mode in ('export', 'train') and hvd.rank() == 0:
        save_model(model, args.logdir)

    
if __name__ == "__main__":
    # 设置需要在命令行中更改的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default = "train")
    parser.add_argument("--logdir", type = str, default = os.path.join(os.path.dirname(__file__), 'log'))
    args = parser.parse_args()
    
    # 执行主代码
    main(args)