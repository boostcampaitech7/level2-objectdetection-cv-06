#### inheritance
_base_ = ['./projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py']

#### hook config
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))

### Running Config
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(  # Multi-processing config
        mp_start_method='fork',  # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend='nccl'),  # Distribution configs
)
vis_backends = [ # Visualization backends.
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ])
log_processor = dict(
    type='LogProcessor',  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.
log_level = 'INFO'  # The level of logging.
# 모델 로드 경로 설정
load_from = './checkpoints/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth'
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
fp16 = dict(loss_scale='dynamic')

### Model config
image_size = (1024, 1024)
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 10
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# 모델의 query_head와 roi_head의 num_classes를 변경
model = dict(
    data_preprocessor=dict(batch_augments=batch_augments),
    query_head=dict(
        num_classes=num_classes  # Use num_classes here
    ),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
)

# 데이터셋 설정
dataset_type = 'CocoDataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data_root = '../../dataset/'

train_cfg = dict(max_epochs=5)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        data_root='../../dataset/',
        ann_file='train_sep.json',
        dataset=dict(pipeline=train_pipeline),
        metainfo=dict(classes=classes)
        ))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
    num_workers=8,
    data_root='../../dataset/',
    ann_file='val_sep.json',
    metainfo=dict(classes=classes)
    ))

val_evaluator = dict(
    _scope_='mmdet',
    ann_file='../../dataset/val_sep.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

test_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
    num_workers=8,
    data_root='../../dataset/',
    ann_file='test.json',
    metainfo=dict(classes=classes)
    ))

test_evaluator = dict(
    _scope_='mmdet',
    ann_file='../../dataset/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    save_results_to_json=True,
    outfile_prefix='./work_dirs/co_dino_trash',
    type='CocoMetric')

