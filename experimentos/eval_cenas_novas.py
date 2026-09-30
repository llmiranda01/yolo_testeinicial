# -*- coding: utf-8 -*-
"""Avalia A e B APENAS nas cenas novas do Agro-vision2 (rajada de 25/09).

O modelo A nunca viu essa rajada; o B treinou em 38 frames dela.
E aqui que o dado novo deve provar seu valor.
"""
import os

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs_treino")
EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_novo.yaml")


def main():
    from ultralytics import YOLO
    out = {}
    for name in ("A_299", "B_353"):
        w = os.path.join(RUNS, name, "weights", "best.pt")
        out[name] = {}
        for split in ("val", "test"):
            r = YOLO(w).val(data=EV, split=split, imgsz=640, batch=8, device=0,
                            workers=0, plots=False, verbose=False,
                            project=os.path.join(RUNS, "_avaliacao_novo"),
                            name="%s_%s" % (name, split), exist_ok=True)
            out[name][split] = (float(r.box.map50), float(r.box.map),
                                float(r.box.mp), float(r.box.mr))
    print("\n@@@ CENAS NOVAS DE 25/09 @@@")
    for split, n in (("val", 11), ("test", 5)):
        print("\n--- %s (%d imagens) ---" % (split, n))
        a, b = out["A_299"][split], out["B_353"][split]
        for i, lab in enumerate(("mAP50", "mAP50-95", "precisao", "recall")):
            d = b[i] - a[i]
            pct = (d / a[i] * 100) if a[i] else float("inf")
            print("   %-9s A=%.4f  B=%.4f  delta=%+.4f (%+.1f%%)" % (lab, a[i], b[i], d, pct))


if __name__ == "__main__":
    main()
