import os
import logging
import tensorflow as tf
import horovod.tensorflow as hvd


def hvd_init():
    hvd.init()
    print("I am worker {}/{}".format(hvd.rank(), hvd.size()))

    # 设置显示的日志级别，用于控制输出哪些信息
    tf.get_logger().setLevel(logging.INFO)

    # 设置gpu信息
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = tf.config.list_physical_devices('GPU')
    
    # 为程序指定gpu
    tf.config.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')
    # TensorFlow 自动选择一个现有且受支持的设备来运行操作，gpu优先级高于cpu
    tf.config.set_soft_device_placement(True)
    
    # 设置cpu运算的核数
    tf.config.threading.set_inter_op_parallelism_threads(8)


class HorovodModel(tf.keras.Model):

    def __init__(self, ):
        super().__init__()
        self.iterations = None
        self.clip_grad_fn = None

    def compile(self,
                optimizer,
                *,
                dynamic_loss_scale,
                run_eagerly=False,
                clip_grad_fn=None):
        # 读取optimizer的迭代次数
        self.iterations = optimizer.iterations
        self.clip_grad_fn = clip_grad_fn

        if dynamic_loss_scale:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer, dynamic=False, initial_scale=1.0)

        super(HorovodModel, self).compile(optimizer=optimizer,
                                          run_eagerly=run_eagerly)

    # tensorflow默认执行动态图，使用tf.function装饰器后，可将这个函数转化为静态图结点进行加速
    @tf.function
    def hvd_broadcast(self, iterations):
        if iterations == 0:
            hvd.broadcast_variables(self.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

    def train_step(self, *, tape, loss):
        iterations = self.iterations.read_value()

        with tape:
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # 用于在tensorboard中显示标量信息
        tf.summary.scalar('loss_scale',
                          tf.math.divide_no_nan(scaled_loss, loss))
        
        # 多机多卡计算梯度
        tape = hvd.DistributedGradientTape(tape)
        vars = self.trainable_variables
        scaled_grads = tape.gradient(scaled_loss, vars)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        
        # 梯度裁减
        if self.clip_grad_fn is not None:
            grads = self.clip_grad_fn(grads, vars)
            
        # 更新参数
        self.optimizer.apply_gradients(zip(grads, vars))

        # 更新iterations的值
        self.iterations.assign(iterations + 1, read_value=False)
        
        # 如果是第0步则需要广播模型参数和optimizer的参数
        self.hvd_broadcast(iterations)

        return {'loss': loss}

    def fit(self,
            dataset,
            epochs,
            logdir,
            summary_freq=100,
        ):
        # 新建一个CheckpointManager用于checkpoint储存，只在主进程上保存
        ckpt = tf.train.Checkpoint(net=self, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=3)

        def save_ckpt():
            ckpt_manager.save(checkpoint_number=self.iterations)

        cpkt_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_ckpt())

        # 打印log
        def print_log(logs):
            print('iter {} loss: {:.6f}'.format(self.iterations.read_value(),
                                                logs['loss']))

        log_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: print_log(logs))

        # 启动tensorboard
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir if hvd.rank() == 0 else '/tmp',
            write_graph=True,
            update_freq=summary_freq)

        if hvd.rank() == 0:
            callbacks = [cpkt_callback, log_callback, tb_callback]
        else:
            callbacks = [log_callback, tb_callback]

        # verbose：日志显示
        # verbose = 0 为不在标准输出流输出日志信息
        # verbose = 1 为输出进度条记录
        # verbose = 2 为每个epoch输出一行记录
        super(HorovodModel, self).fit(
            dataset,
            epochs = epochs,
            initial_epoch = self.iterations // len(dataset),
            verbose = 2,
            callbacks=callbacks,
        )

    # 重新加载logdir中的模型checkpoint
    # Checkpoint只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数
    # 如果需要导出模型（无需源代码也能运行模型）请使用SavedModel
    def restore_or_initialize(self, logdir):
        # 初始化的参数是一系列的键值对，键名可以随意取，值为需要保存的对象
        ckpt = tf.train.Checkpoint(net=self, optimizer=self.optimizer)
        # 定义一个checkpointmanager来管理模型的checkpoint
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            logdir,
            max_to_keep=None,
        )
        # 如果有checkpoint加载，否则_init_fn
        latest = ckpt_manager.restore_or_initialize()
        if latest:
            print('Restore checkpoint from {}'.format(latest))
