from mmengine import Config
from mmengine.runner import set_random_seed

import sys
sys.path.append('..')

# 기본 설정 파일 로드
config_name = 'deformable-detr-refine-twostage_r50_16xb2-50e_coco'
cfg = Config.fromfile(f'../configs/deformable_detr/{config_name}.py')

# 사용자 정의 설정
cfg.load_from = '../checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'

# cfg.train_dataloader.batch_size = 1
cfg.train_dataloader.num_workers = 8

cfg.model.backbone.frozen_stages = -1

# cfg.max_epochs = 12
# cfg.train_cfg.max_epochs = cfg.max_epochs

cfg.metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}

cfg.model.bbox_head.num_classes = len(cfg.metainfo['classes'])

cfg.data_root = '../../dataset'
# cfg.train_dataloader.dataset.dataset.ann_file = 'json/splits/train_fold4.json'
# cfg.train_dataloader.dataset.dataset.data_root = cfg.data_root
# cfg.train_dataloader.dataset.dataset.data_prefix.img = ''
# cfg.train_dataloader.dataset.dataset.metainfo = cfg.metainfo

cfg.train_dataloader.dataset.ann_file = 'json/splits/train_fold4.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = ''
cfg.train_dataloader.dataset.metainfo = cfg.metainfo


cfg.val_dataloader.dataset.ann_file = 'json/splits/val_fold4.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = ''
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'json/splits/val_fold4.json'
cfg.test_evaluator = cfg.val_evaluator

# cfg.default_hooks.checkpoint.interval = 4

# model weights are saved every 10 intervals, up to two weights are saved at the same time, and the saving strategy is auto
cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='auto')

# Interval of reporting indicators
cfg.default_hooks.logger.interval = 1

cfg.device = 'cuda'

set_random_seed(42, deterministic=False)

cfg.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend'),
]

custom_config_name = config_name + '_trash'
cfg.work_dir = f'../work_dirs/{custom_config_name}'

# 설정 파일 저장
config_path = f'../custom_configs/{custom_config_name}.py'
cfg.dump(config_path)

print(f"Custom config saved to {config_path}")