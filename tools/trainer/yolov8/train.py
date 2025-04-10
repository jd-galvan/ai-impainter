from ultralytics import YOLO

model = YOLO("yolov8x.pt")

model.train(
    data="./config.yaml",         # usa tu dataset combinado
    #imgsz=1024,                 # mejora la detección de manchas pequeñas
    epochs=200,
    patience=50,
    batch=8,                    # bajá si ves errores de memoria
    lr0=0.001,
    name="full_dataset_yolov8x9",
    project="runs/detect",
    device=[1],
    dropout=0.3,
    optimizer="SGD",
    weight_decay=0.0001
)
