from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="./config.yaml",         # usa tu dataset combinado
    #imgsz=1024,                 # mejora la detección de manchas pequeñas
    epochs=400,
    patience=100,
    batch=8,                    # bajá si ves errores de memoria
    lr0=0.001,
    name="full_dataset_yolov8n6",
    project="runs/detect",
    device=[1]
)
