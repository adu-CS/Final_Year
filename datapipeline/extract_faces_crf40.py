import os
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def setup_directories(base_out_dir):
    """Creates the output folder structure for testing data."""
    dirs = [
        os.path.join(base_out_dir, 'real'),
        os.path.join(base_out_dir, 'fake')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0], dirs[1]

def process_video(video_path, output_dir, mtcnn, video_id, max_faces=30, frame_skip=15):
    """Extracts faces from a single video using GPU MTCNN."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    faces_extracted = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            save_path = os.path.join(output_dir, f"{video_id}_frame{frame_count}.jpg")

            try:
                face = mtcnn(pil_img, save_path=save_path) 
                if face is not None:
                    faces_extracted += 1
            except Exception as e:
                pass

        frame_count += 1
        
        if faces_extracted >= max_faces:
            break

    cap.release()

def main():
    # --------------------------------------------------------- #
    # CUSTOMIZED PATHS FOR YOUR SYSTEM
    # --------------------------------------------------------- #
    SOURCE_REAL_VIDEOS = r"C:\Users\srikar\crf40dataset\original_sequences\youtube\c40\videos"
    SOURCE_FAKE_VIDEOS = r"C:\Users\srikar\crf40dataset\manipulated_sequences\Deepfakes\c40\videos"
    OUTPUT_BASE_DIR    = r"C:\Users\srikar\ai_engineering_hub\Deepfake_detection_project\crf40_evaluation_data"
    
    # --- THE NEW LIMITS ---
    MAX_VIDEOS_PER_CLASS = 150  # Only process 150 videos per category
    MAX_FACES_PER_VIDEO  = 30   # Max 30 faces per video
    FRAME_INTERVAL       = 15
    # --------------------------------------------------------- #

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {DEVICE}")

    mtcnn = MTCNN(
        image_size=224, 
        margin=20, 
        keep_all=False,       
        select_largest=True, 
        post_process=False,   
        device=DEVICE
    )

    out_real_dir, out_fake_dir = setup_directories(OUTPUT_BASE_DIR)

    # Process Real Videos (Capped at 150)
    print(f"\n--- Extracting Real Faces (Limit: {MAX_VIDEOS_PER_CLASS} videos) ---")
    real_vids = [v for v in os.listdir(SOURCE_REAL_VIDEOS) if v.endswith(('.mp4', '.avi'))]
    real_vids = real_vids[:MAX_VIDEOS_PER_CLASS]  # Apply the cutoff
    
    for vid_name in tqdm(real_vids):
        vid_path = os.path.join(SOURCE_REAL_VIDEOS, vid_name)
        vid_id = vid_name.split('.')[0]
        process_video(vid_path, out_real_dir, mtcnn, vid_id, MAX_FACES_PER_VIDEO, FRAME_INTERVAL)

    # Process Fake Videos (Capped at 150)
    print(f"\n--- Extracting Fake Faces (Limit: {MAX_VIDEOS_PER_CLASS} videos) ---")
    fake_vids = [v for v in os.listdir(SOURCE_FAKE_VIDEOS) if v.endswith(('.mp4', '.avi'))]
    fake_vids = fake_vids[:MAX_VIDEOS_PER_CLASS]  # Apply the cutoff
    
    for vid_name in tqdm(fake_vids):
        vid_path = os.path.join(SOURCE_FAKE_VIDEOS, vid_name)
        vid_id = vid_name.split('.')[0]
        process_video(vid_path, out_fake_dir, mtcnn, vid_id, MAX_FACES_PER_VIDEO, FRAME_INTERVAL)

    print(f"\n[SUCCESS] CRF 40 testing dataset is ready at: {OUTPUT_BASE_DIR}")

if __name__ == '__main__':
    main()