export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="2" python fp_inference.py configs/sample.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 
