import cv2
import torch

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) # Adjust path if necessary

# Define the IP camera URL (Change it to your IP camera URL)
ip_camera_url = 'http:10.102.138.126:8080'

# Start capturing video from the IP camera
cap = cv2.VideoCapture(ip_camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 640x640 for the model
    frame_resized = cv2.resize(frame, (640, 640))

    # Perform inference
    results = model(frame_resized)

    # Check if fire or smoke is detected in the results
    labels = results.names
    for label, confidence, bbox in zip(results.xyxy[0][:, -1].numpy(), results.xyxy[0][:, 4].numpy(), results.xyxy[0][:, :4].numpy()):
        if labels[int(label)] in ['fire', 'smoke']:  # Assuming 'fire' and 'smoke' are the labels in your model
            print(f"{labels[int(label)]} detected with confidence {confidence}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
