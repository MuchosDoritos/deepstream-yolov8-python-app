import time
import queue
import threading
import cv2
import numpy as np
from ultralytics import YOLO

class FPSCounter:
    def __init__(self):
        self.frames = 0
        self.start = time.time()
        self.fps = 0.0
 
    def tick(self):
        self.frames += 1
        now = time.time()
        if now - self.start >= 1.0:
            self.fps = self.frames / (now - self.start)
            self.frames = 0
            self.start = now
            print(f"FPS: {self.fps:.2f}")

# Load YOLO model (engine format)
model = YOLO('./models/rock-paper-scissors.engine')

# Shared queues and control event
frame_queue = queue.Queue(maxsize=12)
result_queue = queue.Queue(maxsize=12)
stop_event = threading.Event()

def run_inference(frame):
    """Run YOLO inference on the frame"""
    try:
        results = model(frame, verbose=False)
        return results[0] if results else None
    except Exception as e:
        print(f"Inference error: {e}")
        return None

def draw_prediction(frame, prediction):
    """Draw bounding boxes and labels on the frame"""
    if prediction is None:
        return frame
    
    annotated_frame = frame.copy()
    
    # Draw bounding boxes and labels
    if hasattr(prediction, 'boxes') and prediction.boxes is not None:
        boxes = prediction.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = model.names[cls] if cls < len(model.names) else f'Class {cls}'
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name}: {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_frame

def show(frame):
    """Display the frame"""
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()

def capture_thread():
    """Capture frames from camera"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 350)
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested FPS: 500, Actual camera FPS: {actual_fps}")

    print("Starting capture thread...")
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    
    cap.release()
    print("Capture thread stopped")

def inference_thread():
    """Run inference on frames"""
    print("Starting inference thread...")
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Run object detection on frame
        prediction = run_inference(frame)
        result = (frame, prediction)
        
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            # Drop oldest result if queue is full
            try:
                result_queue.get_nowait()
                result_queue.put_nowait(result)
            except queue.Empty:
                pass
    
    print("Inference thread stopped")

def render_thread():
    """Render frames with predictions"""
    fps = FPSCounter()
    print("Starting render thread...")
    
    while not stop_event.is_set():
        try:
            frame, prediction = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Draw predictions on frame
        annotated_frame = draw_prediction(frame, prediction)
        
        # Display frame (optional GUI)
        show(annotated_frame)
        fps.tick()
    
    print("Render thread stopped")

def main():
    """Main function to start all threads"""
    print("Starting multithreaded YOLO inference...")
    print("Press 'q' to quit")
    
    # Create and start threads
    threads = []
    
    capture_t = threading.Thread(target=capture_thread, name="CaptureThread")
    inference_t = threading.Thread(target=inference_thread, name="InferenceThread")
    render_t = threading.Thread(target=render_thread, name="RenderThread")
    
    threads.extend([capture_t, inference_t, render_t])
    
    # Start all threads
    for thread in threads:
        thread.start()
        print(f"Started {thread.name}")
    
    try:
        # Wait for threads to complete
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        stop_event.set()
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=2)
    
    # Cleanup
    cv2.destroyAllWindows()
    print("All threads stopped, exiting...")

if __name__ == "__main__":
    main()
