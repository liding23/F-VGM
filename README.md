# FlightVGM-OpenSora1.2 GPU Profiling

## Hardware and Software Requirements
- RTX 3090 GPU
- CUDA 12.1
- Anaconda

## Environment Setup
**Step 1**: In order to avoid library version errors, we exported environment.yml according to the libraries required by OpenSora1.2,   
            and you can install it directly according to this.
```
# create a conda environment following F-VGM/environment.yaml
cd F-VGM
conda env create -f environment.yaml
conda activate F-VGM
```
**Stpe 2**: Install flash-attention for inference acceleration
```
pip install flash-attn --no-build-isolation
```

## Speed Comparison of Dense Inference and Sparse Inference
**Prompt Path**: ./F-VGM/prompts.txt   
**Weight Path**: Weight will be automatically downloaded when you run the inference script  
**Config Path**: ./F-VGM/configs  
>Frames = 64, Resolution = 240p,  
Other settings are OpenSora1.2 default settings.
### Dense Inference
```
# Generate three videos at a time and output the total inference time.
bash dense_inference.sh
```
### Sparse Inference
```
# Generate three videos at a time and output the total inference time.
bash sparse_inference.sh
```
## Generated video Quality Comparison
``` 
# Generated video path
Dense outputs: /F-VGM/logs/dense_samples
Sparse outpus: /F-VGM/logs/sparse_samples

# Clipsim scores
Path: /F-VGM/clip_results/ 
```

