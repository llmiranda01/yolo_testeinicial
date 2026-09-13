from ultralytics import YOLO
import cv2


# Carrega o modelo treinado
model = YOLO("best.pt")


# Abre a câmera
camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print("Camera nao encontrada")
    exit()


while True:

    ret, frame = camera.read()

    if not ret:
        print("Erro lendo camera")
        break


    # Faz a detecção
    results = model(frame)


    # Desenha caixas e nomes
    frame_resultado = results[0].plot()


    # Mostra a imagem
    cv2.imshow("Reconhecimento de vasos", frame_resultado)


    # Aperte Q para fechar
    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()