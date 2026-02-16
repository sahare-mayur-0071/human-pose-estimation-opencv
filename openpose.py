import cv2 as cv
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video')
    parser.add_argument('--thr', default=0.2, type=float)
    parser.add_argument('--width', default=368, type=int)
    parser.add_argument('--height', default=368, type=int)
    return parser.parse_args()

args = get_args()

BODY_PARTS = {
    "Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4,
    "LShoulder":5, "LElbow":6, "LWrist":7, "RHip":8, "RKnee":9,
    "RAnkle":10, "LHip":11, "LKnee":12, "LAnkle":13,
    "REye":14, "LEye":15, "REar":16, "LEar":17, "Background":18
}

POSE_PAIRS = [
    ["Neck","RShoulder"], ["Neck","LShoulder"],
    ["RShoulder","RElbow"], ["RElbow","RWrist"],
    ["LShoulder","LElbow"], ["LElbow","LWrist"],
    ["Neck","RHip"], ["RHip","RKnee"], ["RKnee","RAnkle"],
    ["Neck","LHip"], ["LHip","LKnee"], ["LKnee","LAnkle"],
    ["Neck","Nose"], ["Nose","REye"], ["REye","REar"],
    ["Nose","LEye"], ["LEye","LEar"]
]

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
cap = cv.VideoCapture(args.input if args.input else 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv.dnn.blobFromImage(
        frame, 1.0,
        (args.width, args.height),
        (127.5,127.5,127.5),
        swapRB=True, crop=False
    )

    net.setInput(blob)
    output = net.forward()[:, :19, :, :]

    points = []

    for i in range(len(BODY_PARTS)):
        heatmap = output[0, i]
        _, conf, _, point = cv.minMaxLoc(heatmap)

        x = int(w * point[0] / output.shape[3])
        y = int(h * point[1] / output.shape[2])

        points.append((x,y) if conf > args.thr else None)

    for partA, partB in POSE_PAIRS:
        idA, idB = BODY_PARTS[partA], BODY_PARTS[partB]

        if points[idA] and points[idB]:
            cv.line(frame, points[idA], points[idB], (0,255,0), 3)
            cv.circle(frame, points[idA], 3, (0,0,255), -1)
            cv.circle(frame, points[idB], 3, (0,0,255), -1)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, "%.2f ms" % (t / freq),
               (10,20), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (0,0,0))

    cv.imshow("OpenPose using OpenCV", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
