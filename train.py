import datetime
import itertools
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.model import discriminator, generator
from utils.callbacks import LossHistory
# from utils.dataloader import OurModel_dataset_collate, OurModelDataset

from utils.dataloader import CycleGan_dataset_collate, CycleGanDataset

from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True

    distributed     = False

    fp16            = False

    G_model_A2B_path    = ""
    G_model_B2A_path    = ""
    D_model_A_path      = ""
    D_model_B_path      = ""

    # 设置输入图像的尺寸
    input_shape     = [256, 512]
    

    #  设置训练的总周期数（100）以及初始周期（0）。
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 2

    # 学习率 Init_lr 设置为 2e-4，Min_lr 为其 1%。
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01

    # 选择 Adam 优化器。
    optimizer_type      = "adam"
    momentum            = 0.5
    weight_decay        = 0
    lr_decay_type       = "cos"
    save_period         = 10
    # 训练结果保存路径，记录日志。
    save_dir            = 'logs'
    num_workers         = 2
    photo_save_step     = 50
    annotation_path = "train_lines.txt"


    # 初始化分布式训练环境
    # 初始化分布式训练：torch.cuda.device_count() 获取 GPU 数量，dist.init_process_group() 初始化分布式进程组。
    # rank 和 local_rank 用于获取当前进程的排名和局部排名。
    # 如果不是分布式训练，则直接使用 GPU 或 CPU。
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0


    # 初始化模型
    # 初始化生成器（generator）和判别器（discriminator）模型。
    # CycleGAN 需要两个生成器（A2B 和 B2A）和两个判别器（A 和 B）。

    G_model_A2B = generator()
    G_model_B2A = generator()
    D_model_A   = discriminator()
    D_model_B   = discriminator()

    # 如果指定了预训练模型路径，就加载模型的权重。
    if G_model_A2B_path != '':
        pretrained_dict = torch.load(G_model_A2B_path, map_location=device)
        G_model_A2B.load_state_dict(pretrained_dict)
    if G_model_B2A_path != '':
        pretrained_dict = torch.load(G_model_B2A_path, map_location=device)
        G_model_B2A.load_state_dict(pretrained_dict)
    if D_model_A_path != '':
        pretrained_dict = torch.load(D_model_A_path, map_location=device)
        D_model_A.load_state_dict(pretrained_dict)
    if D_model_B_path != '':
        pretrained_dict = torch.load(D_model_B_path, map_location=device)
        D_model_B.load_state_dict(pretrained_dict)
    

    # 使用交叉熵损失（BCE）和均方误差损失（MSE）作为 CycleGAN 的损失函数。
    BCE_loss = nn.BCEWithLogitsLoss()
    MSE_loss = nn.MSELoss()

    # 记录训练损失
    # 在 local_rank == 0 的机器上初始化 LossHistory，用于记录训练过程中生成器和判别器的损失。
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, [G_model_A2B, G_model_B2A, D_model_A, D_model_B], input_shape=input_shape)
    else:
        loss_history    = None

    # 如果启用了混合精度训练（fp16），则使用 GradScaler 来管理精度的缩放。
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # 将模型切换为训练模式，并根据 GPU 数量进行配置
    G_model_A2B_train = G_model_A2B.train()
    G_model_B2A_train = G_model_B2A.train()
    D_model_A_train = D_model_A.train()
    D_model_B_train = D_model_B.train()

    # 在 GPU 上使用模型
    if Cuda:
        if distributed:
            G_model_A2B_train = G_model_A2B_train.cuda(local_rank)
            G_model_A2B_train = torch.nn.parallel.DistributedDataParallel(G_model_A2B_train, device_ids=[local_rank], find_unused_parameters=True)
            
            G_model_B2A_train = G_model_B2A_train.cuda(local_rank)
            G_model_B2A_train = torch.nn.parallel.DistributedDataParallel(G_model_B2A_train, device_ids=[local_rank], find_unused_parameters=True)
            
            D_model_A_train = D_model_A_train.cuda(local_rank)
            D_model_A_train = torch.nn.parallel.DistributedDataParallel(D_model_A_train, device_ids=[local_rank], find_unused_parameters=True)
            
            D_model_B_train = D_model_B_train.cuda(local_rank)
            D_model_B_train = torch.nn.parallel.DistributedDataParallel(D_model_B_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            cudnn.benchmark = True
            
            G_model_A2B_train = torch.nn.DataParallel(G_model_A2B_train)
            G_model_A2B_train = G_model_A2B_train.cuda()

            G_model_B2A_train = torch.nn.DataParallel(G_model_B2A_train)
            G_model_B2A_train = G_model_B2A_train.cuda()

            D_model_A_train = torch.nn.DataParallel(D_model_A_train)
            D_model_A_train = D_model_A_train.cuda()

            D_model_B_train = torch.nn.DataParallel(D_model_B_train)
            D_model_B_train = D_model_B_train.cuda()
    
    # 准备数据集
    with open(annotation_path) as f:
        lines = f.readlines()
    annotation_lines_A, annotation_lines_B = [], []
    for annotation_line in lines:
        annotation_lines_A.append(annotation_line) if int(annotation_line.split(';')[0]) == 0 else annotation_lines_B.append(annotation_line)
    num_train = max(len(annotation_lines_A), len(annotation_lines_B))

    if local_rank == 0:
        show_config(
            input_shape = input_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
            )

    # 配置优化器
    if True:

        G_optimizer = {
            'adam'  : optim.Adam(itertools.chain(G_model_A2B_train.parameters(), G_model_B2A_train.parameters()), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(itertools.chain(G_model_A2B_train.parameters(), G_model_B2A_train.parameters()), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        D_optimizer_A = {
            'adam'  : optim.Adam(D_model_A_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(D_model_A_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        D_optimizer_B = {
            'adam'  : optim.Adam(D_model_B_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(D_model_B_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]


        # 学习率调节器
        # get_lr_scheduler：这个函数会根据 lr_decay_type（如 "cos"），初始化学习率调度器。学习率调度器会在训练过程中动态调整学习率，通常有几种策略，比如：学习率指数衰减、余弦衰减等。这里用的是余弦衰减（"cos"）。
        # lr_scheduler_func：返回的学习率调度器函数，用于在每个 epoch 开始时调整学习率。
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        # 计算每个 epoch 中的训练步骤数
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")


        # train_dataset   = OurModelDataset(annotation_lines_A, annotation_lines_B, input_shape)
        # CycleGanDataset：这是一个自定义的数据集类（显然是为 CycleGAN 设计的）。
        # 它将两个数据源（annotation_lines_A 和 annotation_lines_B）传递给 CycleGanDataset 类，生成训练用的图像对。
        train_dataset   = CycleGanDataset(annotation_lines_A, annotation_lines_B, input_shape)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
    
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=CycleGan_dataset_collate, sampler=train_sampler)

        # 训练循环
        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)

            # 每个 epoch 开始时，调用该函数根据学习率调度器来调整优化器的学习率。
            set_optimizer_lr(G_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(D_optimizer_A, lr_scheduler_func, epoch)
            set_optimizer_lr(D_optimizer_B, lr_scheduler_func, epoch)

            # 这个函数会执行每个 epoch 的训练过程，具体包括：前向传播、损失计算、反向传播、梯度更新等步骤。
            fit_one_epoch(G_model_A2B_train, G_model_B2A_train, D_model_A_train, D_model_B_train, G_model_A2B, G_model_B2A, D_model_A, D_model_B, loss_history,
                        G_optimizer, D_optimizer_A, D_optimizer_B, BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir, photo_save_step, local_rank)

            if distributed:
                dist.barrier()
