import argparse
import datetime
import os
from tkinter import filedialog
from tkinter import *
from typing import List, Optional, Union
import time
import numpy as np
import torch
import torchvision.ops.boxes as bops
import pygame
import norfair
from norfair import Detection, Paths, Tracker, Video
from distances import frobenius, iou
import cv2  
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

audio = 'sound_warning\mixkit-alarm-clock-beep-988.wav'
def sendWarning():
    pygame.mixer.init()
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()

class YOLO:
    def __init__(self, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model
        try:
            self.model = torch.load('local_model.pt')  # local model
        except:
            raise Exception("Failed to load model")

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections

def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "bbox"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLOv5 inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.2",
    help="YOLOv7 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.40", help="YOLOv5 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="bbox",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO(device=args.device)
time_track =dict()

distance_function = iou if args.track_points == "bbox" else frobenius

distance_threshold = (
    DISTANCE_THRESHOLD_BBOX
    if args.track_points == "bbox"
    else DISTANCE_THRESHOLD_CENTROID
)

tracker = Tracker(
    distance_function=distance_function,
    distance_threshold=distance_threshold,
)
font = cv2.FONT_HERSHEY_SIMPLEX


def main_function(device = 'CAMERA_LAPTOP',url =''):
    
    prev_frame_time = 0
    new_frame_time = 0
    # Initialize frame counters.
    good_frames = 0
    bad_frames = 0

    # Font type.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Colors.
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)

    if device == 'CAMERA_LAPTOP':
        cap = cv2.VideoCapture(0)
    elif device == 'VIDEO':
        try:
            root = Tk()
            file_path = filedialog.askopenfilename()
            root.destroy()
            cap = cv2.VideoCapture(file_path)
        except: return

    elif device == 'CAMERA_PHONE':
        try:
            cap = cv2.VideoCapture(url)
        except:
            print('wrong url')
            return
    # Meta
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    #result = cv2.VideoWriter(name_video,cv2.VideoWriter_fourcc(*'MP4V'),10, size)
    while True:
        ret,frame = cap.read()
        if (ret==False): break
        frame_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        yolo_detections = model(
            frame_0,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            image_size=args.img_size,
            classes=args.classes,
        )
        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=args.track_points
        )

        tracked_objects = tracker.update(detections=detections)
        for i in range(len(tracked_objects)):
                time_track[tracked_objects[i].id] = time_track.get(tracked_objects[i].id, 0) + 1
                if (time_track[tracked_objects[i].id]/int(fps)>5.0):
                    cv2.putText(frame, "Warning", (450, 70), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    sendWarning()
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, "fps: "+fps, (7, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
            norfair.draw_tracked_boxes(frame, tracked_objects)
        cv2.imshow('cam', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

#main_function('CAMERA_LAPTOP')