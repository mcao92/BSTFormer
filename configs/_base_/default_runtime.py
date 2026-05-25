data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

optimizer = dict(
    type="Adam",
    lr=0.0001,
)

loss = dict(type="MSELoss")

runner = dict(max_epochs=20)

checkpoint_config = dict(interval=1)

log_config = dict(interval=60)

save_image_config = dict(interval=100)

eval = dict(
    flag=False,
    interval=1,
)

checkpoints = None
resume = None
