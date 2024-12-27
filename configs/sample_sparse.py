resolution = "240p"
aspect_ratio = "9:16"
num_frames = 64
fps = 24
frame_interval = 1
save_fps = 24
save_dir = "./logs/sparse_samples"
seed = 1024
batch_size = 1
multi_resolution = "STDiT2"
dtype = "fp16"
condition_frame_length = 5
align = 5

do_sparse = True

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False, 
    do_sparse = do_sparse,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None

precompute_text_embeds = False
