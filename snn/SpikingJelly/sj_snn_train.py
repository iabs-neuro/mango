# imports

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

try:
    from torchvision import prototype
except ImportError:
    prototype = None

import numpy as np

from spikingjelly.activation_based.model.train_classify import set_deterministic, seed_worker
from spikingjelly.activation_based.model.tv_ref_classify import transforms, utils

import os
import sys
import time
import datetime
import shutil
import tqdm

from ...data.data_main import Data
from .SJ_snn import SResNetTrainer

def main():
    batch_size = 300
    data_ = Data('cifar10', batch_trn=batch_size, batch_tst=batch_size, force_reload=False)

    params = {
        'data-path': os.path.join(data_.root, '_data', data_.name),
        'batch-size': batch_size,
        'distributed': False,
        'cutmix-alpha': 1.0,
        'model': 'spiking_resnet18',
        'workers': 1,
        'T': 50,
        'train-crop-size': 32,
        'cupy': True,
        'epochs': 1000,
        'lr': 0.2,
        'random-erase': 0.1,
        'label-smoothing': 0.1,
        'momentum': 0.9
        #'resume': 'latest'
    }

    trainer = SResNetTrainer()
    parser = trainer.get_args_parser()
    parser.add_argument('--distributed', type=bool, help="distributed")
    #parser.add_argument('--resume', type=bool, help="resume")

    remove_prev_checkpoint = False #

    args, _ = parser.parse_known_args()

    for argpair in params.items():
        args = parser.parse_args(['--' + str(argpair[0]), str(argpair[1])], namespace=args)

    args = parser.parse_args(['--distributed', params['distributed']], namespace=args)
    #args = parser.parse_args(['--resume', params['resume']], namespace=args)

    print(args)

    # -----------------------------------------------------------------------------
    #                                   Training
    # -----------------------------------------------------------------------------

    if 'resume' not in params or not params['resume']:
        shutil.rmtree('./logs', ignore_errors=True)

    set_deterministic(args.seed, args.disable_uda)

    if args.prototype and prototype is None:
        raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
    if not args.prototype and args.weights:
        raise ValueError("The weights parameter works only in prototype mode. Please pass the --prototype argument.")
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    dataset, dataset_test, train_sampler, test_sampler = trainer.load_CIFAR10(args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
            pass
        else:
            # TODO implement a CrossEntropyLoss to support for probabilities for each class.
            raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                        "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")

        try:
            mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        except AttributeError:
            pass

    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=not args.disable_pinmemory,
        #collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=not args.disable_pinmemory,
        worker_init_fn=seed_worker
    )

    print("Creating model")
    model = trainer.load_model(args, num_classes)
    model.to(device)
    #print(model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    optimizer = trainer.set_optimizer(args, parameters)

    if args.disable_amp:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler()

    lr_scheduler = trainer.set_lr_scheduler(args, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    # 确定目录文件名

    tb_dir = trainer.get_tb_logdir_name(args)
    pt_dir = os.path.join(args.output_dir, 'pt', tb_dir)
    tb_dir = os.path.join(args.output_dir, tb_dir)

    if args.print_logdir:
        print(tb_dir)
        print(pt_dir)
        exit()

    if args.clean:
        if utils.is_main_process():
            if os.path.exists(tb_dir):
                os.remove(tb_dir)
            if os.path.exists(pt_dir):
                os.remove(pt_dir)
            print(f'remove {tb_dir} and {pt_dir}.')

    if utils.is_main_process():
        os.makedirs(tb_dir, exist_ok=args.resume is not None)
        os.makedirs(pt_dir, exist_ok=args.resume is not None)

    if args.resume is not None:
        if args.resume == 'latest':
            checkpoint = torch.load(os.path.join(pt_dir, 'checkpoint_latest.pth'), map_location="cpu")
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"])

        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

        if utils.is_main_process():
            max_test_acc1 = checkpoint['max_test_acc1']
            if model_ema:
                max_ema_test_acc1 = checkpoint['max_ema_test_acc1']

    if utils.is_main_process():
        tb_writer = SummaryWriter(tb_dir, purge_step=args.start_epoch)
        with open(os.path.join(tb_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
            args_txt.write('\n')
            args_txt.write(' '.join(sys.argv))

        max_test_acc1 = -1.
        if model_ema:
            max_ema_test_acc1 = -1.

    if args.test_only:
        if model_ema:
            trainer.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            trainer.evaluate(args, model, criterion, data_loader_test, device=device)

    for epoch in tqdm.tqdm(np.arange(args.start_epoch, args.epochs)):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainer.before_train_one_epoch(args, model, epoch)
        '''
        #==============================
        header = f"Epoch: [{epoch}]"
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, -1, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image = trainer.preprocess_train_sample(args, image)
                print(image.shape)
                print(model)
                print(model(image).shape)
                output = trainer.process_model_output(args, model(image))
                loss = criterion(output, target)
        #==============================
        '''
        train_loss, train_acc1, train_acc5 = trainer.train_one_epoch(model, criterion, optimizer, data_loader, device,
                                                                     epoch, args, model_ema, scaler)
        if utils.is_main_process():
            tb_writer.add_scalar('train_loss', train_loss, epoch)
            tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            tb_writer.add_scalar('train_acc5', train_acc5, epoch)

        lr_scheduler.step()
        trainer.before_test_one_epoch(args, model, epoch)
        test_loss, test_acc1, test_acc5 = trainer.evaluate(args, model, criterion, data_loader_test, device=device)
        if utils.is_main_process():
            tb_writer.add_scalar('test_loss', test_loss, epoch)
            tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            tb_writer.add_scalar('test_acc5', test_acc5, epoch)
        if model_ema:
            ema_test_loss, ema_test_acc1, ema_test_acc5 = trainer.evaluate(args, model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            if utils.is_main_process():
                tb_writer.add_scalar('ema_test_loss', ema_test_loss, epoch)
                tb_writer.add_scalar('ema_test_acc1', ema_test_acc1, epoch)
                tb_writer.add_scalar('ema_test_acc5', ema_test_acc5, epoch)

        if utils.is_main_process():
            save_max_test_acc1 = False
            save_max_ema_test_acc1 = False

            if test_acc1 > max_test_acc1:
                max_test_acc1 = test_acc1
                save_max_test_acc1 = True

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "max_test_acc1": max_test_acc1,
            }
            if model_ema:
                if ema_test_acc1 > max_ema_test_acc1:
                    max_ema_test_acc1 = ema_test_acc1
                    save_max_ema_test_acc1 = True
                checkpoint["model_ema"] = model_ema.state_dict()
                checkpoint["max_ema_test_acc1"] = max_ema_test_acc1
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(pt_dir, "checkpoint_latest.pth"))
            if save_max_test_acc1:
                utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_test_acc1.pth"))
            if model_ema and save_max_ema_test_acc1:
                utils.save_on_master(checkpoint, os.path.join(pt_dir, f"checkpoint_max_ema_test_acc1.pth"))

            if utils.is_main_process() and epoch > 0 and remove_prev_checkpoint:
                os.remove(os.path.join(pt_dir, f"checkpoint_{epoch - 1}.pth"))

        print(f'escape time={(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        #print(args)

if __name__ == '__main__':
    main()
