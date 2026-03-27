import os
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def setup_directories(base_out_dir):
    """Creates the 5th folder structure for Celeb-DF evaluation."""
    dirs = [
        os.path.join(base_out_dir, 'real'),
        os.path.join(base_out_dir, 'fake')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0], dirs[1]

def process_video(video_path, output_dir, mtcnn, video_id, max_faces=30, frame_skip=15):
    """Extracts faces using GPU-based MTCNN detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False

    faces_extracted = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            save_path = os.path.join(output_dir, f"{video_id}_frame{frame_count}.jpg")

            try:
                face = mtcnn(pil_img, save_path=save_path) 
                if face is not None:
                    faces_extracted += 1
            except Exception:
                pass

        frame_count += 1
        if faces_extracted >= max_faces:
            break

    cap.release()
    return True

def main():
    # --------------------------------------------------------- #
    # PATH CONFIGURATION
    # --------------------------------------------------------- #
    BASE_INPUT_DIR  = r"D:\srikar\Celeb-DF-v2"
    TEST_LIST_PATH  = os.path.join(BASE_INPUT_DIR, "List_of_testing_videos.txt")
    OUTPUT_BASE_DIR = r"C:\Users\srikar\ai_engineering_hub\Deepfake_detection_project\celebdf_evaluation_data"
    
    MAX_VIDEOS_PER_CLASS = 150 # Keeping it balanced and manageable
    MAX_FACES_PER_VIDEO  = 30
    FRAME_INTERVAL       = 15
    # --------------------------------------------------------- #

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Running Celeb-DF Extraction on: {DEVICE}")

    mtcnn = MTCNN(image_size=224, margin=20, select_largest=True, post_process=False, device=DEVICE)
    out_real_dir, out_fake_dir = setup_directories(OUTPUT_BASE_DIR)

    # Parse the official test list
    real_count = 0
    fake_count = 0
    
    with open(TEST_LIST_PATH, 'r') as f:
        lines = f.readlines()

    print(f"\n--- Processing Celeb-DF Official Test Split ---")
    for line in tqdm(lines):
        parts = line.strip().split(' ')
        if len(parts) < 2: continue
        
        label = int(parts[0]) # 1 = Real, 0 = Fake
        rel_path = parts[1]
        
        # Determine target and current counts
        if label == 1 and real_count < MAX_VIDEOS_PER_CLASS:
            target_dir = out_real_dir
            is_valid = True
        elif label == 0 and fake_count < MAX_VIDEOS_PER_CLASS:
            target_dir = out_fake_dir
            is_valid = True
        else:
            is_valid = False

        if is_valid:
            abs_video_path = os.path.join(BASE_INPUT_DIR, rel_path)
            vid_id = os.path.basename(rel_path).split('.')[0]
            
            if os.path.exists(abs_video_path):
                success = process_video(abs_video_path, target_dir, mtcnn, vid_id, MAX_FACES_PER_VIDEO, FRAME_INTERVAL)
                if success:
                    if label == 1: real_count += 1
                    else: fake_count += 1
            
        # Stop if we hit our class limits
        if real_count >= MAX_VIDEOS_PER_CLASS and fake_count >= MAX_VIDEOS_PER_CLASS:
            break

    print(f"\n[SUCCESS] Celeb-DF faces ready at: {OUTPUT_BASE_DIR}")
    print(f"Total Videos Processed: {real_count} Real, {fake_count} Fake")

if __name__ == '__main__':
    main()