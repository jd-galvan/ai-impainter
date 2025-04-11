from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="./config.yaml",
    epochs=100,
    patience=15,
    batch=8,
    lr0=0.001,
    name="full_dataset_yolov8x12",
    project="runs/detect",
    device=[0],
    dropout=0.3,
    optimizer="SGD",
    weight_decay=0.0001,
)
