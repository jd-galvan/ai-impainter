import os

from ultralytics import YOLO
model = YOLO("yolov8s.pt")

model.train(data="./config.yaml", epochs=400, device=[1])

