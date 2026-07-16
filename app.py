from fastapi import FastAPI, Response, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import threading
import time
import urllib.request
import shutil

app = FastAPI()

# Enable CORS for decoupled dev environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_logged_in = True  # Toggle login state

# Upload configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global runtime stats for UI integration
runtime_stats = {
    'fps': 0,
    'object_count': 0,
    'human_count': 0,
    'vehicle_count': 0,
    'motion_detected': False,
    'current_emotion': 'Neutral',
    'current_gesture': 'No Hand Detected',
    'detection_log': []  # Holds recent detection events
}

def log_event(event_text):
    """Utility to log events to the runtime stats log."""
    timestamp = time.strftime("%H:%M:%S")
    runtime_stats['detection_log'].insert(0, {'time': timestamp, 'text': event_text})
    if len(runtime_stats['detection_log']) > 10:
        runtime_stats['detection_log'].pop()

# Global settings configuration
detection_settings = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'motion_sensitivity': 1000,
    'show_bounding_boxes': True,
    'motion_display_mode': 'color'  # 'color', 'mask', or 'split'
}

# --- Thread-Safe Camera Manager ---
class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Error: Could not open video source.")
            self.status = False
        else:
            self.status = True
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if not self.status:
                time.sleep(0.1)
                continue
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            time.sleep(0.015)  # Restrict thread loop speed (approx 60fps cap)

    def read(self):
        with self.read_lock:
            if not self.status or not self.grabbed:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.started = False
        if self.stream.isOpened():
            self.stream.release()

# Global instance of CameraStream
global_camera = None

def get_camera_stream():
    global global_camera
    if global_camera is None:
        global_camera = CameraStream().start()
    return global_camera

def get_fallback_frame(text="Camera Not Available"):
    # Generate black frame with warning text
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

# --- Global Model Loader ---
class ModelLoader:
    _yolo_net = None
    _yolo_classes = None
    _yolo_output_layers = None
    
    _face_cascade = None
    _eye_cascade = None
    _fer_detector = None
    
    _lock = threading.Lock()

    @classmethod
    def get_yolo(cls):
        with cls._lock:
            if cls._yolo_net is None:
                weights_path = os.path.join("python_Scripts", "yolov3.weights")
                cfg_path = os.path.join("python_Scripts", "yolov3.cfg")
                coco_path = os.path.join("python_Scripts", "coco.names")
                
                # Check if full YOLOv3 weights exist
                if not os.path.exists(weights_path):
                    print(f"Full YOLOv3 weights not found at {weights_path}.")
                    # Check if yolov3-tiny weights exist
                    tiny_weights_path = os.path.join("python_Scripts", "yolov3-tiny.weights")
                    tiny_cfg_path = os.path.join("python_Scripts", "yolov3-tiny.cfg")
                    
                    if not os.path.exists(tiny_weights_path):
                        print("YOLOv3-tiny weights not found. Downloading YOLOv3-tiny (33MB) fallback...")
                        cls.download_file("https://pjreddie.com/media/files/yolov3-tiny.weights", tiny_weights_path)
                    
                    if not os.path.exists(tiny_cfg_path):
                        print("Downloading YOLOv3-tiny configuration...")
                        cls.download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", tiny_cfg_path)
                    
                    weights_path = tiny_weights_path
                    cfg_path = tiny_cfg_path
                    print("Initialized YOLOv3-tiny model.")
                else:
                    print("Initialized full YOLOv3 model.")

                cls._yolo_net = cv2.dnn.readNet(weights_path, cfg_path)
                layer_names = cls._yolo_net.getLayerNames()
                cls._yolo_output_layers = [layer_names[i - 1] for i in cls._yolo_net.getUnconnectedOutLayers()]
                
                with open(coco_path, "r") as f:
                    cls._yolo_classes = [line.strip() for line in f.readlines()]
            return cls._yolo_net, cls._yolo_classes, cls._yolo_output_layers

    @classmethod
    def download_file(cls, url, dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            print(f"Downloading {url} to {dest_path}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                block_size = 1024 * 8
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except:
                    pass
            raise e

    @classmethod
    def get_cascades(cls):
        with cls._lock:
            if cls._face_cascade is None:
                cls._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                cls._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            return cls._face_cascade, cls._eye_cascade

    @classmethod
    def get_fer(cls):
        with cls._lock:
            if cls._fer_detector is None:
                try:
                    from fer import FER
                    cls._fer_detector = FER(mtcnn=False)
                except Exception as e:
                    print(f"Could not load FER module: {e}. Falling back to cascade heuristic.")
                    cls._fer_detector = "fallback"
            return cls._fer_detector

# --- Hand Gesture Logic ---
def detect_hand_gesture(roi):
    """
    Detect hand gestures in ROI using HSV thresholding and convexity defects.
    Counts fingers and returns (gesture_label, mask, annotated_roi).
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Generic skin-color range in HSV
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Filter operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No Hand Detected", mask, roi
    
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 5000:
        return "No Hand Detected", mask, roi
    
    # Draw contour
    cv2.drawContours(roi, [contour], -1, (0, 255, 255), 2)
    
    # Hull & Defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Check angles to exclude non-finger gaps
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                
                if angle <= 90 and d > 3000:
                    count_defects += 1
                    cv2.circle(roi, far, 5, (0, 0, 255), -1)
                    cv2.line(roi, start, end, (255, 0, 0), 2)
            
            # Label gesture
            if count_defects == 0:
                return "Fist / One Finger (0-1)", mask, roi
            elif count_defects == 1:
                return "Victory / Peace (2)", mask, roi
            elif count_defects == 2:
                return "Three Fingers (3)", mask, roi
            elif count_defects == 3:
                return "Four Fingers (4)", mask, roi
            elif count_defects == 4:
                return "Open Hand / Hello (5)", mask, roi
            else:
                return "Hand Detected", mask, roi
    return "Hand Detected", mask, roi


# --- Pipeline Detection Helpers ---

def process_object(frame):
    net, classes, output_layers = ModelLoader.get_yolo()
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    height, width, _ = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(detection[4] * scores[class_id])  # Adjusted formula to prevent conf > 1.0

            if confidence > detection_settings['confidence_threshold']:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, detection_settings['confidence_threshold'], detection_settings['nms_threshold'])
    runtime_stats['object_count'] = len(indices)

    if detection_settings['show_bounding_boxes'] and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            conf = confidences[i]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({round(conf * 100, 1)}%)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if np.random.rand() < 0.05:
                log_event(f"Detected: {label} ({round(conf * 100, 1)}%)")
    return frame

def process_human(frame):
    net, classes, output_layers = ModelLoader.get_yolo()
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    human_count = 0
    height, width, _ = frame.shape
    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(detection[4] * scores[class_id])

            if confidence > detection_settings['confidence_threshold'] and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, detection_settings['confidence_threshold'], detection_settings['nms_threshold'])
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            if detection_settings['show_bounding_boxes']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            human_count += 1
    
    runtime_stats['human_count'] = human_count
    cv2.putText(frame, f'Humans detected: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if human_count > 0 and np.random.rand() < 0.05:
        log_event(f"Detected {human_count} human(s) in frame")
    return frame

def process_vehicle(frame):
    net, classes, output_layers = ModelLoader.get_yolo()
    vehicle_classes = {2, 3, 5, 7}  # car, motorbike, bus, truck
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    vehicle_count = 0
    height, width, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(detection[4] * scores[class_id])

            if confidence > detection_settings['confidence_threshold'] and (class_id in vehicle_classes):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, detection_settings['confidence_threshold'], detection_settings['nms_threshold'])
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            if detection_settings['show_bounding_boxes']:
                label = classes[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({round(confidences[i]*100, 1)}%)", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            vehicle_count += 1
    
    runtime_stats['vehicle_count'] = vehicle_count
    cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if vehicle_count > 0 and np.random.rand() < 0.05:
        log_event(f"Detected {vehicle_count} vehicle(s) in frame")
    return frame

def process_movement(frame, mog_subtractor):
    frame_resized = cv2.resize(frame, (640, 480))
    blurred_frame = cv2.GaussianBlur(frame_resized, (5, 5), 0)
    
    foreground_mask = mog_subtractor.apply(blurred_frame, learningRate=0.01)
    _, thresholded_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > detection_settings['motion_sensitivity']:
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    runtime_stats['motion_detected'] = motion_detected
    
    if motion_detected and np.random.rand() < 0.05:
        log_event("Movement detected!")

    display_mode = detection_settings['motion_display_mode']
    if display_mode == 'mask':
        output_frame = clean_mask
    elif display_mode == 'split':
        mask_color = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        output_frame = np.hstack((frame_resized, mask_color))
    else:
        output_frame = frame_resized
    return output_frame

def process_emotion(frame):
    detector = ModelLoader.get_fer()
    face_cascade, eye_cascade = ModelLoader.get_cascades()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if detector != "fallback" and detector is not None:
            face_roi = frame[y:y+h, x:x+w]
            emotions = detector.detect_emotions(face_roi)
            if emotions:
                dominant_emotion, score = detector.top_emotion(face_roi)
                if dominant_emotion:
                    emotion_label = f"{dominant_emotion.capitalize()} ({round(score*100, 1)}%)"
                    runtime_stats['current_emotion'] = dominant_emotion.capitalize()
                    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if np.random.rand() < 0.05:
                        log_event(f"Emotion detected: {dominant_emotion}")
            else:
                cv2.putText(frame, "Analyzing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            mouth_region = roi_gray[int(h / 2):h, :]
            _, mouth_thresh = cv2.threshold(mouth_region, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            smiling = any(cv2.contourArea(c) > 300 for c in contours)

            if len(eyes) >= 2 and smiling:
                emotion = "Happy"
            elif len(eyes) < 2 and not smiling:
                emotion = "Sad"
            else:
                emotion = "Neutral"

            runtime_stats['current_emotion'] = emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

def process_sign(frame):
    height, width, _ = frame.shape
    roi_x_start = int(width * 0.5)
    roi_y_start = 100
    roi_w = 300
    roi_h = 300

    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_start + roi_w, roi_y_start + roi_h), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand inside box", (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    roi = frame[roi_y_start:roi_y_start+roi_h, roi_x_start:roi_x_start+roi_w]
    gesture, mask, annotated_roi = detect_hand_gesture(roi)
    
    frame[roi_y_start:roi_y_start+roi_h, roi_x_start:roi_x_start+roi_w] = annotated_roi
    
    runtime_stats['current_gesture'] = gesture
    cv2.putText(frame, f"Sign: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if gesture != "No Hand Detected" and np.random.rand() < 0.05:
        log_event(f"Gesture recognized: {gesture}")

    if detection_settings['motion_display_mode'] == 'mask':
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame[roi_y_start:roi_y_start+roi_h, roi_x_start:roi_x_start+roi_w] = mask_rgb
    return frame


# --- API Routes for REST & MJPEG Video Streaming ---

@app.get('/api/stats')
async def get_stats():
    return JSONResponse(runtime_stats)

@app.get('/api/settings')
async def get_settings():
    return JSONResponse(detection_settings)

@app.post('/api/settings')
async def update_settings(request: Request):
    data = await request.json()
    if 'confidence_threshold' in data:
        detection_settings['confidence_threshold'] = max(0.1, min(1.0, float(data['confidence_threshold'])))
    if 'nms_threshold' in data:
        detection_settings['nms_threshold'] = max(0.1, min(1.0, float(data['nms_threshold'])))
    if 'motion_sensitivity' in data:
        detection_settings['motion_sensitivity'] = max(100, int(data['motion_sensitivity']))
    if 'show_bounding_boxes' in data:
        detection_settings['show_bounding_boxes'] = bool(data['show_bounding_boxes'])
    if 'motion_display_mode' in data:
        detection_settings['motion_display_mode'] = str(data['motion_display_mode'])
    return JSONResponse({'status': 'success', 'settings': detection_settings})

@app.post('/api/upload')
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    # Clean filename
    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ['.', '_', '-']]).strip()
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    
    # Save file asynchronously
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return JSONResponse({'status': 'success', 'file_path': file_path})

@app.get('/api/video_feed/{mode}')
def video_feed(mode: str, source: str = 'webcam'):
    """
    Serves the MJPEG video feed stream. FastAPI automatically spawns a separate
    thread to execute this synchronous generator, keeping the main async event loop
    free and fully responsive.
    """
    def generate_frames():
        if source == 'webcam':
            camera = get_camera_stream()
        else:
            camera = cv2.VideoCapture(source)
            
        last_time = time.time()
        mog_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        try:
            while True:
                if source == 'webcam':
                    success, frame = camera.read()
                else:
                    success, frame = camera.read()
                    if not success:
                        # Looping pre-recorded video
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, frame = camera.read()
                
                if not success:
                    if source == 'webcam':
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + get_fallback_frame("Webcam unavailable") + b'\r\n')
                        time.sleep(0.1)
                        continue
                    else:
                        break
                
                # FPS Calculation
                curr_time = time.time()
                time_diff = curr_time - last_time
                fps = int(1.0 / time_diff) if time_diff > 0 else 0
                last_time = curr_time
                runtime_stats['fps'] = fps

                # Inference Processing
                try:
                    if mode == 'object':
                        processed_frame = process_object(frame)
                    elif mode == 'human':
                        processed_frame = process_human(frame)
                    elif mode == 'vehicle':
                        processed_frame = process_vehicle(frame)
                    elif mode == 'movement':
                        processed_frame = process_movement(frame, mog_subtractor)
                    elif mode == 'emotion':
                        processed_frame = process_emotion(frame)
                    elif mode == 'sign':
                        processed_frame = process_sign(frame)
                    else:
                        processed_frame = frame
                except Exception as e:
                    print(f"Error processing frame in mode {mode}: {e}")
                    processed_frame = frame

                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Yield control to prevent thread starvation
                time.sleep(0.01)
        finally:
            if source != 'webcam':
                camera.release()

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# --- Serving React Single-Page Application (SPA) ---

# Serve files uploaded in cloud environments
@app.get('/static/uploads/{filename}')
async def serve_upload(filename: str):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Serve general static project files (logos, fallback graphics)
@app.get('/static/{path:path}')
async def serve_project_static(path: str):
    return send_from_directory('static', path)

def send_from_directory(directory, path):
    file_path = os.path.join(directory, path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Simple content-type detection
    media_type = "application/octet-stream"
    if path.endswith(".png"):
        media_type = "image/png"
    elif path.endswith(".jpg") or path.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif path.endswith(".mp4"):
        media_type = "video/mp4"
    elif path.endswith(".css"):
        media_type = "text/css"
    elif path.endswith(".js"):
        media_type = "application/javascript"
    elif path.endswith(".html"):
        media_type = "text/html"
        
    return StreamingResponse(open(file_path, "rb"), media_type=media_type)

# Mount compiled React frontend files
if os.path.exists('frontend/dist'):
    app.mount('/', StaticFiles(directory='frontend/dist', html=True), name='frontend')
else:
    @app.get('/')
    async def root_fallback():
        return Response(
            "React frontend build not found. Please compile frontend using 'npm run build' inside the 'frontend' directory.",
            media_type="text/plain"
        )
