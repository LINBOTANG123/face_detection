from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


def detect_face_single(vid_path: Path):
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    cap = cv2.VideoCapture(str(vid_path))
    frames = 0
    full_face_detected = 0
    unstable_count = 0
    
    brightness_threshold_dark = 50
    brightness_threshold_light = 200
    unstable_threshold = 0.7
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return f"Unable to read video file {vid_path.name}"
    
    average_brightness = (brightness_threshold_light + brightness_threshold_dark) / 2
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            if np.mean(mag) > unstable_threshold:
                unstable_count += 1
                
            prev_gray = gray
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    face_region = frame[y:y+h, x:x+w]
                    
                    if not face_region.size:
                        continue
                    average_brightness = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY).mean()
                    rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    landmarks = face_mesh.process(rgb_face_region).multi_face_landmarks
                    
                    if landmarks and len(landmarks) > 0:
                        full_face_detected += 1
    
    frac_full_face_detected = (full_face_detected / frames) if frames > 0 else 0
    result = f"{vid_path.name},{frac_full_face_detected:.2f},{average_brightness},{unstable_count/frames:.2f}"
    cap.release()
    cv2.destroyAllWindows()
    
    return result

def detect_faces_files(file_paths, result_file: Path):
    header = "filename,frac full face detected,average brightness,frac unstable"
    print(header)
    
    with result_file.open('w') as f:
        f.write(f"{header}\n")
        for file_path in file_paths:
            result = detect_face_single(file_path)
            f.write(f"{result}\n")

# file_paths = [Path('/Users/linbotang/Documents/video/dataset/Beta_Test_Videos/1694209618941$$image_picker_C282160F_38EF_40E9_9722_620C6A1C8EBD_21021_0000033FA019E7D471590239214__DEF147D0_462E_4B63_9D1A_30643A5DDBC9A6BFE990_CF7F_43FF_82CB_9A138F7E9FFD.mp4'), Path('/Users/linbotang/Documents/video/dataset/Beta_Test_Videos/1694274607403$$image_picker_07CEBE31_1A63_4752_AB90_EF04D3A94BA7_1080_00000017638EA99371596734951__E7EC5537_9A93_4DB9_A76D_4632704B284D04BEF005_0BE3_4C99_AE68_4D25DBE5631A.mp4')]
# print(file_paths)
# output_file = Path('result.csv')
# detect_faces_files(file_paths, output_file)