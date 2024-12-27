export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="1" python sparse_inference.py configs/sample_sparse.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 

python clip_score.py \
  --video-folder ./logs/sparse_samples \
  --text-file ./prompts.txt \
  --output-folder ./clip_results \
  --output-filename clip_scores_sparse_1.txt