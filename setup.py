# import cv2
# from ultralytics import YOLO
# import cvzone
# import math

# def detect_fire(stream_url, model):
#     # Initialize video capture with the stream URL
#     cap = cv2.VideoCapture(stream_url)

#     if not cap.isOpened():
#         print("Error: Could not open stream.")
#         return

#     print("Starting stream feed. Press 'q' to quit.")
#     frame_counter =  0
#     while True:
#         success, img  = cap.read()
#         frame_counter +=  1

#         # Only process every  30th frame
#         if frame_counter %  5 !=  0:
#             continue

#         results = model(img, stream=True)
        
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 w, h = x2-x1, y2-y1

#                 cls = box.cls[0]
#                 name = str(cls)
#                 print(name)
#                 cvzone.putTextRect(img, f'{name}', (max(0,x1), max(35,y1)), scale =  1.5)

#         print(img.shape)
#         cv2.imshow("Image", img)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     try:
#     # Initialize the YOLO model
#         model = YOLO('C:\\You Love Me\\best.pt')  # Use the path to your trained model weights
#         print("Model Loaded Successfully")
#     except Exception as e:
#         print(e)
#     # Stream URL from the IP camera
#     stream_url = 'rtsp://10.102.138.191:8080/h264_opus.sdp'  # Replace with your actual stream URL

#     # Start real-time detection
#     detect_fire(stream_url, model)

import cv2
from ultralytics import YOLO
import cvzone

def detect_fire(stream_url, model):
    # Initialize video capture with the stream URL
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open stream.")
        return

    print("Starting stream feed. Press 'q' to quit.")
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Resize the frame to the model's expected input size (assuming 640x640 for this example)
        img_resized = cv2.resize(img, (640, 640))

        # Perform inference
        results = model(img_resized)

        # Process detections
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            for detection in results:
                x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
                confidence = detection['confidence']
                cls_id = detection['class']
                label = model.names[cls_id]  # Assuming 'model.names' contains class labels

                # Check if the detected label is 'fire' or 'smoke'
                if label in ['fire', 'smoke']:  # Adjust based on your model's classes
                    # Print only if 'fire' or 'smoke' is detected
                    print(f"{label} detected with confidence {confidence}")
                    cvzone.putTextRect(img, f'{label}', (x1, y1-10), scale=1.5, thickness=2, colorR=(0,255,0), offset=0)
        else:
            print("Unexpected results structure:", type(results))

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Initialize the YOLO model
        model = YOLO('C:\\You Love Me\\best.pt')  # Use the path to your trained model weights
        print("Model Loaded Successfully")
    except Exception as e:
        print(e)

    # Stream URL from the IP camera
    stream_url = 'rtsp://10.102.138.193:8080/h264_opus.sdp'  # Replace with your actual stream URL

    # Start real-time detection
    detect_fire(stream_url, model)
