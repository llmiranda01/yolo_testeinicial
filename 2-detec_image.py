from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("foto.jpeg", save=True)



print("[+] Concluído, chorou pae")