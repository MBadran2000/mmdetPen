auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'pipeline',
    ])
data_root = '/scratch/dr/m.badran/pengwin/train/input/images/x-ray/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr_config = dict(
    policy='step',
    step=[
        48,
        66,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333)
metainfo = dict(
    classes=(
        'SA',
        'LI',
        'RI',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            60,
            220,
            20,
        ),
        (
            20,
            60,
            220,
        ),
    ])
model = dict(
    backbone=dict(
        base_width=4,
        depth=101,
        frozen_stages=1,
        groups=64,
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_64x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=3,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=3,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=120,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=120,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=120)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=120,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=120)),
    type='MaskRCNN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        '/scratch/dr/m.badran/pengwin/mmdetection_utils/Val_Config_90.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/scratch/dr/m.badran/pengwin/train/input/images/x-ray/',
        metainfo=dict(
            classes=(
                'SA',
                'LI',
                'RI',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    60,
                    220,
                    20,
                ),
                (
                    20,
                    60,
                    220,
                ),
            ]),
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                448,
                448,
            ), type='Resize'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(epsilon=0.001, type='CustomNegLogTransform'),
            dict(
                convert=True,
                lower=0.01,
                type='CustomWindowTransform',
                upper=0.95),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='PengwinMetric'),
    dict(
        ann_file=
        '/scratch/dr/m.badran/pengwin/mmdetection_utils/Val_Config_90.json',
        backend_args=None,
        format_only=False,
        metric=[
            'bbox',
        ],
        type='CocoMetric'),
]
test_pipeline = [
    dict(imdecode_backend='pillow', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        448,
        448,
    ), type='Resize'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(epsilon=0.001, type='CustomNegLogTransform'),
    dict(convert=True, lower=0.01, type='CustomWindowTransform', upper=0.95),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=60000, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file=
        '/scratch/dr/m.badran/pengwin/mmdetection_utils/Train_Config_90.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/scratch/dr/m.badran/pengwin/train/input/images/x-ray/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'SA',
                'LI',
                'RI',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    60,
                    220,
                    20,
                ),
                (
                    20,
                    60,
                    220,
                ),
            ]),
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        448,
                        448,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(epsilon=0.001, type='CustomNegLogTransform'),
            dict(
                convert=True,
                lower=0.01,
                type='CustomWindowTransform',
                upper=0.99),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
                transforms=[
                    dict(clip_limit=(
                        1,
                        4,
                    ), p=0.5, type='CLAHE'),
                    dict(p=0.5, type='InvertImg'),
                    dict(
                        p=1,
                        transforms=[
                            dict(blur_limit=(
                                3,
                                5,
                            ), type='GaussianBlur'),
                            dict(blur_limit=(
                                3,
                                5,
                            ), type='MotionBlur'),
                            dict(blur_limit=5, type='MedianBlur'),
                        ],
                        type='OneOf'),
                    dict(
                        p=1,
                        transforms=[
                            dict(alpha=(
                                0.2,
                                0.5,
                            ), type='Sharpen'),
                            dict(alpha=(
                                0.2,
                                0.5,
                            ), type='Emboss'),
                        ],
                        type='OneOf'),
                    dict(
                        p=1,
                        transforms=[
                            dict(
                                multiplier=(
                                    0.9,
                                    1.1,
                                ),
                                type='MultiplicativeNoise'),
                            dict(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                type='HueSaturationValue',
                                val_shift_limit=20),
                            dict(
                                brightness_limit=(
                                    -0.4,
                                    0.2,
                                ),
                                contrast_limit=(
                                    -0.4,
                                    0.2,
                                ),
                                type='RandomBrightnessContrast'),
                        ],
                        type='OneOf'),
                    dict(scale=0.1, type='RandomToneCurve'),
                    dict(
                        p=1,
                        transforms=[
                            dict(type='RandomShadow'),
                        ],
                        type='OneOf'),
                ],
                type='Albu'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(imdecode_backend='pillow', type='LoadImageFromFile'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(keep_ratio=True, scales=[
        (
            448,
            448,
        ),
    ], type='RandomChoiceResize'),
    dict(epsilon=0.001, type='CustomNegLogTransform'),
    dict(convert=True, lower=0.01, type='CustomWindowTransform', upper=0.99),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
        transforms=[
            dict(clip_limit=(
                1,
                4,
            ), p=0.5, type='CLAHE'),
            dict(p=0.5, type='InvertImg'),
            dict(
                p=1,
                transforms=[
                    dict(blur_limit=(
                        3,
                        5,
                    ), type='GaussianBlur'),
                    dict(blur_limit=(
                        3,
                        5,
                    ), type='MotionBlur'),
                    dict(blur_limit=5, type='MedianBlur'),
                ],
                type='OneOf'),
            dict(
                p=1,
                transforms=[
                    dict(alpha=(
                        0.2,
                        0.5,
                    ), type='Sharpen'),
                    dict(alpha=(
                        0.2,
                        0.5,
                    ), type='Emboss'),
                ],
                type='OneOf'),
            dict(
                p=1,
                transforms=[
                    dict(multiplier=(
                        0.9,
                        1.1,
                    ), type='MultiplicativeNoise'),
                    dict(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        type='HueSaturationValue',
                        val_shift_limit=20),
                    dict(
                        brightness_limit=(
                            -0.4,
                            0.2,
                        ),
                        contrast_limit=(
                            -0.4,
                            0.2,
                        ),
                        type='RandomBrightnessContrast'),
                ],
                type='OneOf'),
            dict(scale=0.1, type='RandomToneCurve'),
            dict(p=1, transforms=[
                dict(type='RandomShadow'),
            ], type='OneOf'),
        ],
        type='Albu'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        '/scratch/dr/m.badran/pengwin/mmdetection_utils/Val_Config_90.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/scratch/dr/m.badran/pengwin/train/input/images/x-ray/',
        metainfo=dict(
            classes=(
                'SA',
                'LI',
                'RI',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    60,
                    220,
                    20,
                ),
                (
                    20,
                    60,
                    220,
                ),
            ]),
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                448,
                448,
            ), type='Resize'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(epsilon=0.001, type='CustomNegLogTransform'),
            dict(
                convert=True,
                lower=0.01,
                type='CustomWindowTransform',
                upper=0.95),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='PengwinMetric'),
    dict(
        ann_file=
        '/scratch/dr/m.badran/pengwin/mmdetection_utils/Val_Config_90.json',
        backend_args=None,
        format_only=False,
        metric=[
            'bbox',
        ],
        type='CocoMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/scratch/dr/m.badran/pengwin/EXP_Nawar/mask_rcnn_x101V3/'
