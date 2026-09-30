# -*- coding: utf-8 -*-
"""Avalia os modelos A (299 img) e B (353 img) nas MESMAS imagens retidas.

O conjunto comum e o valid (63) e o test (30) do dataset antigo: nenhum dos
dois modelos treinou nessas imagens, portanto a comparacao e justa.
"""
import csv, json, os

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
RUNS = os.path.join(PROJ, "runs_treino")
EVAL = os.path.join(BASE, "eval_comum.yaml")
CLASSES = ["erva_daninha", "planta"]

MODELOS = [
    ("A_299", "299 img (206 train) - antes do Agro-vision2"),
    ("B_353", "353 img (244 train) - com Agro-vision2"),
]


def metricas(r):
    b = r.box
    out = {"mAP50": float(b.map50), "mAP50_95": float(b.map),
           "precisao": float(b.mp), "recall": float(b.mr), "por_classe": {}}
    try:
        f1 = b.f1
        out["f1"] = float(sum(f1) / len(f1))
    except Exception:
        out["f1"] = None
    idx = list(getattr(b, "ap_class_index", range(len(CLASSES))))
    for i, c in enumerate(idx):
        p, rc, ap50, ap = b.class_result(i)[:4]
        out["por_classe"][CLASSES[int(c)]] = {
            "precisao": float(p), "recall": float(rc),
            "mAP50": float(ap50), "mAP50_95": float(ap)}
    return out


def curva(d):
    """Le results.csv: epochs rodadas e melhor epoch por mAP50-95."""
    p = os.path.join(d, "results.csv")
    if not os.path.isfile(p):
        return {}
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return {}
    key = next((k for k in rows[0] if "mAP50-95" in k), None)
    best_i, best_v = 0, -1.0
    for i, r in enumerate(rows):
        try:
            v = float(r[key])
        except (TypeError, ValueError, KeyError):
            continue
        if v > best_v:
            best_v, best_i = v, i
    return {"epochs_rodadas": len(rows), "melhor_epoch": best_i + 1,
            "melhor_mAP50_95_no_valid_proprio": round(best_v, 4)}


def main():
    from ultralytics import YOLO

    res = {}
    for name, desc in MODELOS:
        d = os.path.join(RUNS, name)
        w = os.path.join(d, "weights", "best.pt")
        if not os.path.isfile(w):
            print("!! faltando: %s" % w)
            continue
        print("\n" + "=" * 74)
        print("AVALIANDO %s (%s)" % (name, desc))
        print("=" * 74)
        res[name] = {"desc": desc, "pesos": w, "treino": curva(d)}
        for split, n in (("val", 63), ("test", 30)):
            print("\n-- %s: split=%s (%d imagens comuns) --" % (name, split, n))
            m = YOLO(w)
            r = m.val(data=EVAL, split=split, imgsz=640, batch=8, device=0,
                      plots=True, project=os.path.join(RUNS, "_avaliacao"),
                      name="%s_%s" % (name, split), exist_ok=True, verbose=False)
            res[name][split] = metricas(r)

    with open(os.path.join(RUNS, "comparacao.json"), "w") as f:
        json.dump(res, f, indent=2)

    if len(res) < 2:
        print("\nso um modelo disponivel, sem comparacao")
        return

    a, b = res["A_299"], res["B_353"]
    for split, n in (("val", 63), ("test", 30)):
        print("\n" + "=" * 74)
        print("COMPARACAO - %s comum (%d imagens, nenhum modelo treinou nelas)"
              % (split.upper(), n))
        print("=" * 74)
        print("  %-14s %10s %10s %10s %9s" % ("metrica", "A (299)", "B (353)", "delta", "delta %"))
        print("  " + "-" * 58)
        for k, lab in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"),
                       ("precisao", "precisao"), ("recall", "recall"), ("f1", "F1")):
            va, vb = a[split].get(k), b[split].get(k)
            if va is None or vb is None:
                continue
            d = vb - va
            pct = (d / va * 100) if va else float("nan")
            print("  %-14s %10.4f %10.4f %+10.4f %+8.1f%%" % (lab, va, vb, d, pct))
        print()
        for c in CLASSES:
            pa, pb = a[split]["por_classe"].get(c), b[split]["por_classe"].get(c)
            if not pa or not pb:
                continue
            print("  [%s]" % c)
            for k, lab in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"),
                           ("precisao", "precisao"), ("recall", "recall")):
                d = pb[k] - pa[k]
                print("     %-11s %10.4f %10.4f %+10.4f" % (lab, pa[k], pb[k], d))

    print("\n" + "=" * 74)
    print("TREINO")
    print("=" * 74)
    for name in ("A_299", "B_353"):
        t = res[name]["treino"]
        print("  %-8s epochs=%s  melhor_epoch=%s  melhor_mAP50-95(valid proprio)=%s"
              % (name, t.get("epochs_rodadas"), t.get("melhor_epoch"),
                 t.get("melhor_mAP50_95_no_valid_proprio")))


if __name__ == "__main__":
    main()
