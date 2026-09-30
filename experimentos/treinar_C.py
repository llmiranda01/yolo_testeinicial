# -*- coding: utf-8 -*-
"""Treino C: identico ao B, mas com patience folgada.

Diagnostico: a media movel do mAP50-95 do B ainda subia no epoch 106 quando
o patience=20 disparou. O metrica oscila muito porque o valid tem so 74
imagens, entao 20 epochs sem novo recorde absoluto acontece por ruido.
patience=60 deixa o treino usar os 200 epochs, igualando o orcamento do A
(que rodou 190 epochs efetivos) e removendo esse confundidor da comparacao.
"""
import os, time, json

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)

HP = dict(epochs=200, patience=60, batch=8, imgsz=640, workers=2, cache=False,
          device=0, seed=0, deterministic=True, amp=True, val=True, plots=True,
          project=os.path.join(PROJ, "runs_treino"), exist_ok=False)


def main():
    from ultralytics import YOLO
    print("TREINO C_353_p60 - mesmo dataset do B, patience=60")
    t0 = time.time()
    m = YOLO(os.path.join(PROJ, "yolo11n.pt"))
    m.train(data=os.path.join(BASE, "B_353.yaml"), name="C_353_p60", **HP)
    dt = (time.time() - t0) / 60
    print("\nC_353_p60 concluido em %.1f min (%d epochs)" % (dt, int(m.trainer.epoch) + 1))


if __name__ == "__main__":
    main()
