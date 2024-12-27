export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="2" python dense_inference.py configs/sample_dense.py \
  --num-frames 64 --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt 

CUDA_VISIBLE_DEVICES="2" python clip_score.py \
  --video-folder ./logs/dense_samples \
  --text-file ./prompts.txt \
  --output-folder ./clip_results \
  --output-filename clip_scores_dense.txt


