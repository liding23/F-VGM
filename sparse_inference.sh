export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="1" python sparse_inference.py configs/sample_sparse.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 


python clip_score.py \
    --dir_videos "logs/sparse_samples" \
    --dir_prompts "prompts.txt" \
    --dir_results "clip_results" \
    --metric "clip_score"