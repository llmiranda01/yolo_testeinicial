# -*- coding: utf-8 -*-
"""Desempenho imagem por imagem nas 63 imagens de validacao comuns.

O val() agregado esconde ONDE o modelo mudou. Aqui casamos deteccao com
ground-truth por IoU para contar TP/FP/FN em cada imagem e listar quais
imagens o modelo B melhorou ou piorou.
"""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
RUNS = os.path.join(PROJ, "runs_treino")
LISTA = os.path.join(BASE, "listas", "A_valid.txt")
CLASSES = ["erva_daninha", "planta"]
CONF, IOU_T = 0.25, 0.50


def ler_gt(img):
    lp = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    lp = os.path.splitext(lp)[0] + ".txt"
    out = []
    if not os.path.isfile(lp):
        return out
    for ln in open(lp, encoding="utf-8"):
        p = ln.split()
        if len(p) == 5:
            out.append((int(p[0]), *[float(x) for x in p[1:]]))
    return out


def xywhn_to_xyxy(b, w, h):
    _, cx, cy, bw, bh = b
    return (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def avaliar(model, imgs):
    """Retorna {imagem: {tp, fp, fn, gt}} casando por IoU dentro da classe."""
    import numpy as np
    res = {}
    for i in range(0, len(imgs), 8):
        lote = imgs[i:i + 8]
        preds = model.predict(lote, conf=CONF, imgsz=640, device=0, verbose=False)
        for img, pr in zip(lote, preds):
            h, w = pr.orig_shape
            gt = [(g[0], xywhn_to_xyxy(g, w, h)) for g in ler_gt(img)]
            det = []
            if pr.boxes is not None and len(pr.boxes):
                cls = pr.boxes.cls.cpu().numpy().astype(int)
                xyxy = pr.boxes.xyxy.cpu().numpy()
                cf = pr.boxes.conf.cpu().numpy()
                order = np.argsort(-cf)
                det = [(int(cls[j]), tuple(xyxy[j])) for j in order]
            usados = set()
            tp = 0
            for dc, db in det:
                melhor, melhor_iou = -1, 0.0
                for gi, (gc, gb) in enumerate(gt):
                    if gi in usados or gc != dc:
                        continue
                    v = iou(db, gb)
                    if v > melhor_iou:
                        melhor, melhor_iou = gi, v
                if melhor >= 0 and melhor_iou >= IOU_T:
                    usados.add(melhor)
                    tp += 1
            res[img] = {"tp": tp, "fp": len(det) - tp, "fn": len(gt) - tp, "gt": len(gt)}
    return res


def resumo(d):
    tp = sum(v["tp"] for v in d.values())
    fp = sum(v["fp"] for v in d.values())
    fn = sum(v["fn"] for v in d.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precisao": p, "recall": r, "f1": f1}


def main():
    from ultralytics import YOLO
    imgs = [l.strip() for l in open(LISTA, encoding="utf-8") if l.strip()]
    print("%d imagens de validacao comuns | conf=%.2f IoU=%.2f" % (len(imgs), CONF, IOU_T))

    out = {}
    for name in ("A_299", "B_353", "C_353_p60"):
        w = os.path.join(RUNS, name, "weights", "best.pt")
        if not os.path.isfile(w):
            print("!! faltando %s" % w)
            return
        out[name] = avaliar(YOLO(w), imgs)
        s = resumo(out[name])
        print("\n%s  TP=%d FP=%d FN=%d | precisao=%.4f recall=%.4f F1=%.4f"
              % (name, s["tp"], s["fp"], s["fn"], s["precisao"], s["recall"], s["f1"]))

    a, b = out["B_353"], out["C_353_p60"]
    print("\n" + "=" * 74)
    print("MUDANCAS POR IMAGEM (erros = FP + FN)")
    print("=" * 74)
    dif = []
    for img in imgs:
        ea = a[img]["fp"] + a[img]["fn"]
        eb = b[img]["fp"] + b[img]["fn"]
        dif.append((eb - ea, img, a[img], b[img]))
    melhorou = sorted([d for d in dif if d[0] < 0])
    piorou = sorted([d for d in dif if d[0] > 0], reverse=True)
    igual = [d for d in dif if d[0] == 0]
    print("  imagens que MELHORARAM: %d | PIORARAM: %d | iguais: %d"
          % (len(melhorou), len(piorou), len(igual)))

    for titulo, lst in (("MAIORES GANHOS de C sobre B", melhorou[:8]), ("MAIORES PERDAS de C sobre B", piorou[:8])):
        if not lst:
            continue
        print("\n  %s:" % titulo)
        for d, img, va, vb in lst:
            print("    %-16s gt=%2d | A tp=%2d fp=%2d fn=%2d -> B tp=%2d fp=%2d fn=%2d  (erros %+d)"
                  % (os.path.basename(img), va["gt"], va["tp"], va["fp"], va["fn"],
                     vb["tp"], vb["fp"], vb["fn"], d))

    with open(os.path.join(RUNS, "analise_valid.json"), "w") as f:
        json.dump({k: {"resumo": resumo(v), "por_imagem": {os.path.basename(i): m
                                                           for i, m in v.items()}}
                   for k, v in out.items()}, f, indent=2)
    print("\n-> %s" % os.path.join(RUNS, "analise_valid.json"))


if __name__ == "__main__":
    main()
