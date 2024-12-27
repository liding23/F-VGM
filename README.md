# The Algorithm of F-VGM on GPU

## Hardware and Software Requirements
- NVIDIA RTX 3090 GPU
- CUDA 12.1
- Anaconda

## Environment Setup

> You can also refer to the installation guide in [OpenSora](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation).

**Step 1**: In order to avoid library version errors, we export `environment.yaml` according to the libraries required by [OpenSora 1.2](https://github.com/hpcaitech/Open-Sora), and you can install it directly according to this.
```bash
# create a conda environment following F-VGM/environment.yaml
cd F-VGM
conda env create -f environment.yaml
conda activate F-VGM
```
**Stpe 2**: Install flash-attention for inference acceleration
```bash
pip install flash-attn --no-build-isolation
```

## Speed Comparison of Dense Inference and Sparse Inference
**Prompt Path**: `./F-VGM/prompts.txt`   
**Weight Path**: Weight will be automatically downloaded when you run the inference script  
**Config Path**: `./F-VGM/configs`
>Frames = 64, Resolution = 240p,  
Other settings are OpenSora1.2 default settings.
### Dense Inference
```bash
# Generate three videos at a time and output the total inference time.
./dense_inference.sh
```
### Sparse Inference
```bash
# Generate three videos at a time and output the total inference time.
./sparse_inference.sh
```
## Generated video Quality Comparison
```bash 
# Generated video path
Dense outputs: /F-VGM/logs/dense_samples
Sparse outputs: /F-VGM/logs/sparse_samples

# Clipsim scores
Path: /F-VGM/clip_results/ 
```

