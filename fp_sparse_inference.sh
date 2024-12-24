export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="1" python fp_sparse_inference.py configs/sample_sparse.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 
