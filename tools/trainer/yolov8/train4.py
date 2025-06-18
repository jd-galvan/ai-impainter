from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="./config4.yaml",
    epochs=100,
    patience=20,
    batch=8,
    lr0=0.001,
    name="full_dataset_yolov8x10_17",
    project="runs/detect",
    device=[0],
    dropout=0.3,
    optimizer="SGD",
    weight_decay=0.0001,
    erasing=0.0, # Evitar borrar zonas que puedan confundirse con manchas
    scale=0.8,
    hsv_s=0.9,
    hsv_v=0.6,
    hsv_h=0.4,
    degrees=0.45
)
