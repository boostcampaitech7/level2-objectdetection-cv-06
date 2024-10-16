from mmengine import Config
from mmengine.runner import set_random_seed

import sys
sys.path.append('..')

# 기본 설정 파일 로드
cfg = Config.fromfile('../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')

# 사용자 정의 설정
cfg.load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# cfg.train_dataloader.batch_size = 1
cfg.train_dataloader.num_workers = 8

cfg.model.backbone.frozen_stages = -1

cfg.max_epochs = 1
cfg.train_cfg.max_epochs = cfg.max_epochs

cfg.metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}


cfg.data_root = '../../dataset'
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

config_name = 'faster-rcnn_r50_fpn_1x_trash'
cfg.work_dir = f'../work_dirs/{config_name}'

# 설정 파일 저장
config_path = f'../custom_configs/{config_name}.py'
cfg.dump(config_path)

print(f"Custom config saved to {config_path}")