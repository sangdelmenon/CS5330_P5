# Sangeeth Deleep Menon
# CS5330 Project 5 - Extension: Live Video Digit Recognition
# Spring 2026
#
# Opens the webcam and classifies the digit shown inside a green ROI box in
# real time. The 28x28 preprocessed patch is shown in the top-left corner so
# you can see exactly what the network receives.
#
# Controls:
#   Q  - quit
#   S  - save a screenshot to live_capture.png

import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from task1 import MyNetwork


# loads the trained MNIST model from disk
def load_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# converts a BGR OpenCV ROI to the 28x28 normalised tensor the network expects
def preprocess_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # MNIST is white-on-black; invert if the ROI looks light-on-dark
    if resized.mean() > 128:
        resized = 255 - resized
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(Image.fromarray(resized)).unsqueeze(0)
    return tensor, resized


# draws an overlay showing the prediction and confidence on the frame
def draw_overlay(frame, x1, y1, x2, y2, pred, confidence, processed_28):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = 'Digit: {}  ({:.1f}%)'.format(pred, confidence * 100)
    cv2.putText(frame, label, (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # show the 28x28 input patch magnified in the top-left corner
    thumb = cv2.resize(processed_28, (84, 84), interpolation=cv2.INTER_NEAREST)
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    frame[10:94, 10:94] = thumb_bgr
    cv2.rectangle(frame, (9, 9), (95, 95), (255, 255, 0), 1)
    cv2.putText(frame, '28x28 input', (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)


def run_live(model_path='mnist_model.pth', camera_index=0):
    model = load_model(model_path)
    print('Model loaded from', model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Error: cannot open camera {}.'.format(camera_index))
        return

    print('Live digit recognition — Q to quit, S to save screenshot.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame.')
            break

        h, w = frame.shape[:2]
        # square ROI centred in the frame, half the shorter dimension
        roi_size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        x1 = cx - roi_size // 2
        x2 = cx + roi_size // 2
        y1 = cy - roi_size // 2
        y2 = cy + roi_size // 2

        roi = frame[y1:y2, x1:x2]
        tensor, processed = preprocess_roi(roi)

        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax(dim=1).item()
        confidence = torch.exp(output).max().item()

        draw_overlay(frame, x1, y1, x2, y2, pred, confidence, processed)

        cv2.imshow('Live Digit Recognition  (Q=quit  S=save)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('live_capture.png', frame)
            print('Screenshot saved to live_capture.png')

    cap.release()
    cv2.destroyAllWindows()


def main(argv):
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'
    camera_idx = int(argv[2]) if len(argv) > 2 else 0
    run_live(model_path, camera_idx)


if __name__ == '__main__':
    main(sys.argv)
