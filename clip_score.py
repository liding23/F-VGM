import torch
import clip
from PIL import Image
import cv2
import os
import argparse
from opensora_base.opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

def get_clip_score(video_path, text):
    # Use CUDA as the fixed device
    device = "cuda"
    
    # Load the pre-trained CLIP model
    model, preprocess = clip.load('ViT-B/32', device=device)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables to store total score and frame count
    total_score = 0
    frame_count = 0

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Preprocess the image and tokenize the text
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([text]).to(device)
        
        # Generate embeddings for the image and text
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
        
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate the cosine similarity to get the CLIP score
        clip_score = torch.matmul(image_features, text_features.T).item()
        
        # Accumulate the score and increment frame count
        total_score += clip_score
        frame_count += 1
    
    # Release the video capture object
    cap.release()

    # Calculate the average score
    average_score = total_score / frame_count if frame_count > 0 else 0
    
    return average_score

def calculate_clip_scores(video_folder, text_file, output_folder, output_filename):
    # Use CUDA as the fixed device
    device = "cuda"
    
    # Read the content of the text file (each line will be a separate prompt)
    with open(text_file, 'r') as file:
        text_lines = [line.strip() for line in file.readlines()]
    
    # Get all video files in the folder and sort them alphabetically
    video_paths = sorted(
        [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.lower().endswith(('.mp4', '.avi', '.mov'))]
    )
    
    # Check if the number of videos matches the number of prompts
    if len(video_paths) != len(text_lines):
        raise ValueError(f"The number of videos ({len(video_paths)}) does not match the number of prompts ({len(text_lines)}).")
    
    total_score = 0
    video_count = 0
    results = []

    # Loop through each video and its corresponding prompt
    for video_path, text in zip(video_paths, text_lines):
        score = get_clip_score(video_path, text)
        total_score += score
        video_count += 1
        results.append(f"{os.path.basename(video_path)}: {score}")
    
    # Calculate the overall average score
    average_score = total_score / video_count if video_count > 0 else 0
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file path
    output_file = os.path.join(output_folder, output_filename)
    
    # Save the results to a file, including the average score
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
        f.write(f"\n\nAverage CLIP Score for all videos: {average_score:.4f}")
    
    return average_score

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate CLIP scores for videos in a folder.")
    
    # Adding required arguments for video folder, text file, output folder, and output filename
    parser.add_argument("--video-folder", type=str, required=True, help="Path to the folder containing video files.")
    parser.add_argument("--text-file", type=str, required=True, help="Path to the text file containing the descriptions.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder where results will be saved.")
    parser.add_argument("--output-filename", type=str, required=True, help="Name of the output result file.")
    
    args = parser.parse_args()

    # Calculate CLIP scores and save results
    average_score = calculate_clip_scores(args.video_folder, args.text_file, args.output_folder, args.output_filename)
    print(f"[Average CLIPSIM metric] : {average_score:.4f}")
    print(f"Results saved to {os.path.join(args.output_folder, args.output_filename)}")

if __name__ == "__main__":
    main()
