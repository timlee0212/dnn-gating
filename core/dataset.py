import torch.nn as nn
from timm.data import create_dataset, create_loader, FastCollateMixup, Mixup, AugMixDataset
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


# A wrapper of timm's implementation
def createTrainLoader(dataset_name, config, data_config):
    augs_config = config.Data.augs

    num_aug_splits = 0
    if augs_config.aug_splits > 0:
        assert augs_config.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = augs_config.aug_splits
    # create the train and eval datasets
    dataset_train = create_dataset(
        dataset_name, root=config.Data.path, split="train", is_training=True,
        download=config.Data.download,
        batch_size=config.Trainer.batch_size)

    # setup mixup / cutmix
    mixup_conf = augs_config.mixup
    collate_fn = None
    mixup_fn = None
    mixup_active = mixup_conf.mixup_alpha > 0 or mixup_conf.cutmix_alpha > 0. or mixup_conf.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=mixup_conf.mixup_alpha, cutmix_alpha=mixup_conf.cutmix_alpha,
            cutmix_minmax=mixup_conf.cutmix_minmax,
            prob=mixup_conf.mixup_prob, switch_prob=mixup_conf.mixup_switch_prob, mode=mixup_conf.mixup_mode,
            label_smoothing=augs_config.smoothing, num_classes=config.Data.num_classes)
        if config.Trainer.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = augs_config.train_interpolation
    if augs_config.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=config.Trainer.batch_size,
        is_training=True,
        use_prefetcher=config.Trainer.prefetcher,
        no_aug=augs_config.no_aug,
        re_prob=augs_config.random_earse.reprob,
        re_mode=augs_config.random_earse.remode,
        re_count=augs_config.random_earse.recount,
        re_split=augs_config.random_earse.resplit,
        scale=augs_config.scale,
        ratio=augs_config.ratio,
        hflip=augs_config.hflip,
        vflip=augs_config.vflip,
        color_jitter=augs_config.color_jitter,
        auto_augment=augs_config.autoaug,
        num_aug_repeats=augs_config.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=config.Data.num_workers,
        distributed=config.Experiment.dist,
        collate_fn=collate_fn,
        pin_memory=config.Data.pin_mem
    )

    loader_train.mixup_fn = mixup_fn

    if augs_config.loss.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=augs_config.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if augs_config.loss.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=augs_config.loss.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif augs_config.smoothing:
        if augs_config.loss.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=augs_config.smoothing,
                                               target_threshold=augs_config.loss.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=augs_config.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()

    return loader_train, train_loss_fn


def createValLoader(dataset_name, config, data_config):
    dataset_eval = create_dataset(
        dataset_name, root=config.Data.path, split="val", is_training=False,
        download=config.Data.download,
        batch_size=config.Trainer.val_batch_size)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=config.Trainer.batch_size,
        is_training=False,
        use_prefetcher=config.Trainer.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=config.Data.num_workers,
        distributed=config.Experiment.dist,
        crop_pct=data_config['crop_pct'],
        pin_memory=config.Data.pin_mem,
    )

    eval_loss_fn = nn.CrossEntropyLoss()

    return loader_eval, eval_loss_fn
