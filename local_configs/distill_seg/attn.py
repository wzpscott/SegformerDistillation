_base_ = [
    '../_base_/models/distill.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
log_config = dict(  
    interval=50, 
    hooks=[
        dict(type='TensorboardLoggerHook') 
        # dict(type='TextLoggerHook')
    ])
work_dir = './work_dirs/attn'

model = dict(
        distillation = dict(
        layers=[
            ['backbone.block1.1.norm2','backbone.block1.2.norm2',[32,64],3],
            ['backbone.block2.1.norm2','backbone.block2.7.norm2',[64,128],3],
            ['backbone.block3.1.norm2','backbone.block3.26.norm2',[160,320],3],
            ['backbone.block4.1.norm2','backbone.block4.2.norm2',[256,512],3]
        ],
        weights_init_strategy='equal',
        parse_mode='regular',
        use_attn=False
    ),
    s_pretrain = './pretrained/mit_b0.pth',
    t_pretrain = './pretrained/segformer.b4.512x512.ade.160k.pth',
)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9,0.999), weight_decay=0.01,
                )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=8)
evaluation = dict(interval=2000, metric='mIoU')  