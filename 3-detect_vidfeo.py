from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="video.mp4", #Diretório do vídeo
    save=True, #Salvar ou não o arquivo
    device=0 #Para utilizar a nossa GPU em vez de processador
)