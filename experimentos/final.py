# -*- coding: utf-8 -*-
"""Avaliacao final dos 3 modelos nos MESMOS conjuntos retidos."""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
RUNS = os.path.join(PROJ, "runs_treino")
EV_ANTIGO = os.path.join(BASE, "eval_comum.yaml")   # 63 val / 30 test (cenas antigas)
EV_NOVO = os.path.join(BASE, "eval_novo.yaml")      # 11 val /  5 test (rajada 25/09)

MOD = [("A_299", "206 train, patience=20 (rodou 190 ep)"),
       ("B_353", "244 train, patience=20 (parou em 86 ep)"),
       ("C_353_p60", "244 train, patience=60 (rodou 200 ep)")]


def main():
    from ultralytics import YOLO
    res = {}
    for name, desc in MOD:
        w = os.path.join(RUNS, name, "weights", "best.pt")
        if not os.path.isfile(w):
            print("!! faltando %s" % w); continue
        res[name] = {"desc": desc}
        for tag, yml in (("antigo", EV_ANTIGO), ("novo", EV_NOVO)):
            for split in ("val", "test"):
                r = YOLO(w).val(data=yml, split=split, imgsz=640, batch=8, device=0,
                                workers=0, plots=False, verbose=False,
                                project=os.path.join(RUNS, "_final"),
                                name="%s_%s_%s" % (name, tag, split), exist_ok=True)
                res[name]["%s_%s" % (tag, split)] = {
                    "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                    "precisao": float(r.box.mp), "recall": float(r.box.mr)}

    with open(os.path.join(RUNS, "final.json"), "w") as f:
        json.dump(res, f, indent=2)

    conj = [("antigo_val", "CENAS ANTIGAS - val (63 img)"),
            ("antigo_test", "CENAS ANTIGAS - test (30 img)"),
            ("novo_val", "CENAS NOVAS 25/09 - val (11 img)"),
            ("novo_test", "CENAS NOVAS 25/09 - test (5 img)")]
    print("\n@@@ TABELA FINAL @@@")
    for k, titulo in conj:
        print("\n" + titulo)
        print("  %-11s %9s %10s %9s %8s" % ("modelo", "mAP50", "mAP50-95", "precisao", "recall"))
        print("  " + "-" * 50)
        for name, _ in MOD:
            if name not in res: continue
            m = res[name][k]
            print("  %-11s %9.4f %10.4f %9.4f %8.4f"
                  % (name, m["mAP50"], m["mAP50_95"], m["precisao"], m["recall"]))

    # media macro dos 4 conjuntos, ponderada por n de imagens
    print("\nMEDIA PONDERADA POR N DE IMAGENS (63+30+11+5 = 109)")
    peso = {"antigo_val": 63, "antigo_test": 30, "novo_val": 11, "novo_test": 5}
    print("  %-11s %9s %10s %9s %8s" % ("modelo", "mAP50", "mAP50-95", "precisao", "recall"))
    print("  " + "-" * 50)
    for name, _ in MOD:
        if name not in res: continue
        tot = sum(peso.values())
        agg = {}
        for met in ("mAP50", "mAP50_95", "precisao", "recall"):
            agg[met] = sum(res[name][k][met] * w for k, w in peso.items()) / tot
        print("  %-11s %9.4f %10.4f %9.4f %8.4f"
              % (name, agg["mAP50"], agg["mAP50_95"], agg["precisao"], agg["recall"]))


if __name__ == "__main__":
    main()
