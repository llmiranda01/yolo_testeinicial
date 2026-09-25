from ultralytics import YOLO

model = YOLO("runs/detect/agrovision_v3/weights/best.pt")

results = model.predict(
    source="test/images",
    conf=0.4,
    save=True
)

print("Teste concluÃ­do!")