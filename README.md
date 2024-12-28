# The Algorithm of F-VGM on GPU

## Hardware and Software Requirements
- NVIDIA RTX 3090 GPU
- CUDA 12.1
- Anaconda

## Environment Setup

> You can also refer to the installation guide in [OpenSora](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation).

**Step 1**: In order to avoid library version errors, we export `environment.yaml` according to the libraries required by [OpenSora 1.2](https://github.com/hpcaitech/Open-Sora), and you can install it directly according to this.
```bash
# create a conda environment following environment.yaml
conda env create -f environment.yaml
conda activate F-VGM
```
**Stpe 2**: Install flash-attention for inference acceleration
```bash
pip install flash-attn --no-build-isolation
```

## Running the Code
**Prompt Path**: `prompts.txt`   
**Weight Path**: Weight will be automatically downloaded when you run the inference script  
**Config Path**: `configs/`
>Frames = 64, Resolution = 240p,  
Other settings are OpenSora1.2 default settings.
### Inference
```bash
# Generate three videos at a time and output the total inference time.
./dense_inference.sh
./sparse_inference.sh
```

### Results
Then, the program will output the average generation time for each video and the evaluation results of video quality (using CLIPSIM metrics), which will be saved in `clip_results/`. 

```bash
# Take dense inference as example
[Average Latency]: Inference Time for a video is 32.03 seconds
[Average CLIPSIM metric] : 0.3033
Results saved to ./clip_results/clip_scores_dense.txt
```