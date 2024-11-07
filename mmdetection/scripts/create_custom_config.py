from mmengine import Config
from mmengine.runner import set_random_seed

import sys
sys.path.append('..')

# 기본 설정 파일 로드
config_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco_nomask'
cfg = Config.fromfile(f'../projects/CO-DETR/configs/codino/{config_name}.py')

# 사전 학습된 모델 가중치 파일 경로 설정
cfg.load_from = '../checkpoints/co_dino_ms59_gamma3_img1380_12.pth'

# 데이터 로더 설정
cfg.train_dataloader.batch_size = 1
cfg.train_dataloader.num_workers = 8

# 백본 네트워크의 모든 레이어를 학습 가능하도록 설정
cfg.model.backbone.frozen_stages = -1

# 학습 에폭 수 설정
cfg.max_epochs = 2
cfg.train_cfg.max_epochs = cfg.max_epochs

# 클래스 정보 설정
cfg.metainfo = {
    'classes': ("General trash", "Paper", "Paper pack", "Metal", "Glass",
                "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
}

# 데이터셋 경로 및 설정
cfg.data_root = '../../dataset'
cfg.train_dataloader.dataset.dataset.ann_file = 'json/merged_train_pseudo.json'
cfg.train_dataloader.dataset.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.dataset.data_prefix.img = ''
cfg.train_dataloader.dataset.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'json/splits/val_fold4.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = ''
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

cfg.val_evaluator.ann_file = cfg.data_root + '/' + 'json/splits/val_fold4.json'
cfg.test_evaluator = cfg.val_evaluator

# 체크포인트 저장 설정
cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='auto')

# 로깅 간격 설정
cfg.default_hooks.logger.interval = 1

# 학습 장치 설정
cfg.device = 'cuda'

# 학습률 스케줄러 설정
cfg.param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            1,
        ],
        type='MultiStepLR'),
]

# 랜덤 시드 설정
set_random_seed(42, deterministic=False)

# 시각화 백엔드 설정
cfg.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend'),
]

# 작업 디렉토리 및 설정 파일 이름 설정
custom_config_name = config_name + '_trash'
cfg.work_dir = f'../work_dirs/{custom_config_name}'

# 설정 파일 저장
config_path = f'../custom_configs/{custom_config_name}.py'
cfg.dump(config_path)

print(f"Custom config saved to {config_path}")