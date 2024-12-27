export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="2" python dense_inference.py configs/sample_dense.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 

python clip_score.py \
    --dir_videos "logs/dense_samples" \
    --dir_prompts "prompts.txt" \
    --dir_results "clip_results"\
    --metric "clip_score"