import argparse
import cv2
from ultralytics import YOLO


parser = argparse.ArgumentParser(description='Run Yolov8 inference on a video')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.7, help='Iou threshold for NMS')
parser.add_argument('--max_det', type=int, default=300, help='Number of detection')
parser.add_argument('--augment', type=bool,default=True, help='improves model prediction robustness with the cost of inference speed')
parser.add_argument('--vid_stride', type=int, default=1, help='skips the video frames 1 means infernce on everyframe')
parser.add_argument('--agnostic_nms', type=bool, default=False, help='skips the video frames 1 means infernce on everyframe')




args = parser.parse_args()

model = YOLO('datasets\Crime_best.pt')
# @title Default title text
Cap_obj=cv2.VideoCapture("Clerk bombarded by 4 gunmen during robbery at SW Houston gas station.mp4")
print("check==",Cap_obj.isOpened())

fps = int(Cap_obj.get(cv2.CAP_PROP_FPS))
width = int(Cap_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(Cap_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter("anotated_video\inference_video.mp4",fourcc,fps,(width,height))

while(Cap_obj.isOpened()):
    ret, frame = Cap_obj.read()
    if ret:
        # Apply inference with the provided parameters
        results = model.predict(frame, augment=args.augment, conf=args.conf, iou=args.iou, max_det=args.max_det, vid_stride=args.vid_stride,agnostic_nms=args.agnostic_nms)
        annotated_frame = results[0].plot() # Render the frame with annotations

        output.write(annotated_frame)
    else:
        break


Cap_obj.release()
output.release()
cv2.destroyAllWindows()