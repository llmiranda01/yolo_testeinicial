# -*- coding: utf-8 -*-
"""Treina os modelos A (299 img) e B (353 img) com hiperparametros identicos.

A unica variavel entre os dois treinos e o conjunto de treino, para que a
diferenca nas metricas seja atribuivel ao Agro-vision2 e nao ao acaso.
Calibrado para GTX 1650 (4 GB VRAM) / 7,8 GB RAM.
"""
import os, sys, time, json

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
WEIGHTS = os.path.join(PROJ, "yolo11n.pt")
RUNS = os.path.join(PROJ, "runs_treino")

# identicos nos dois treinos - so o 'data' muda
HP = dict(
    epochs=200,
    patience=20,
    batch=8,          # conservador para 4 GB de VRAM
    imgsz=640,
    workers=2,        # RAM de 7,8 GB e apertada
    cache=False,      # nao cachear: preserva RAM
    device=0,
    seed=0,
    deterministic=True,
    amp=True,         # precisao mista: ~40% menos VRAM
    val=True,
    plots=True,
    project=RUNS,
    exist_ok=False,
)

EXPS = [
    ("A_299", os.path.join(BASE, "A_299.yaml"), "dataset antes do Agro-vision2 (206 train)"),
    ("B_353", os.path.join(BASE, "B_353.yaml"), "dataset atual com Agro-vision2 (244 train)"),
]


def main():
    from ultralytics import YOLO
    import torch

    print("torch %s | cuda %s | %s" % (
        torch.__version__, torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print("VRAM livre %.2f GB / total %.2f GB" % (free / 1e9, total / 1e9))
    print("hiperparametros: " + ", ".join("%s=%s" % (k, v) for k, v in HP.items()
                                          if k not in ("project", "exist_ok")))

    only = sys.argv[1] if len(sys.argv) > 1 else None
    resumo = {}
    for name, data, desc in EXPS:
        if only and only != name:
            continue
        print("\n" + "=" * 74)
        print("TREINO %s - %s" % (name, desc))
        print("=" * 74)
        t0 = time.time()
        model = YOLO(WEIGHTS)
        try:
            model.train(data=data, name=name, **HP)
        except torch.cuda.OutOfMemoryError:
            print("!! VRAM estourou com batch=%d - refazendo com batch=4" % HP["batch"])
            torch.cuda.empty_cache()
            hp = dict(HP, batch=4, name=name + "_b4")
            model = YOLO(WEIGHTS)
            model.train(data=data, **hp)
        dt = time.time() - t0
        rd = model.trainer.save_dir
        resumo[name] = {"dir": str(rd), "minutos": round(dt / 60, 1),
                        "epochs_rodadas": int(model.trainer.epoch) + 1}
        print("\n%s concluido em %.1f min (%d epochs) -> %s"
              % (name, dt / 60, int(model.trainer.epoch) + 1, rd))

    with open(os.path.join(RUNS, "resumo_treinos.json"), "w") as f:
        json.dump(resumo, f, indent=2)
    print("\n" + json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
