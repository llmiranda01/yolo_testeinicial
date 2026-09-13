from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=200, # Ciclos de treinamento (rodando 200x)
    imgsz=640, # Tamanho das imagens
    batch=8, # Quantas imagens por vez
    workers = 0,
    patience = 0,
    device = 0,
    optimizer = "AdamW",  # explicito, senao optimizer="auto" ignora o lr0 abaixo
    lr0 = 0.0015,         # LR inicial mais conservador pra nao destruir os pesos pre-treinados
    cos_lr = True,        # decaimento suave (cosseno) em vez do linear padrao
    close_mosaic = 40,    # desliga mosaico nas ultimas 40 epochs (padrao 10) -> convergencia final em imagens limpas
    degrees = 10,         # NOVO: rotacao de +/-10 graus (padrao 0.0 = nenhuma). Robo ve os vasos em angulos variados
    hsv_v = 0.5,          # mais variacao de brilho (padrao 0.4) -> robustez a iluminacao diferente do dataset
    name = "yolo26n_final_aug"
)
print("Treinamento finalizado!")