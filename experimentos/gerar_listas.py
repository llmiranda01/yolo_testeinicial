# -*- coding: utf-8 -*-
"""Regenera os YAMLs e as listas dos experimentos A e B.

As listas guardam caminhos ABSOLUTOS das imagens, entao precisam ser
regeneradas em cada maquina. Rode este script logo apos clonar o repo:

    python experimentos/gerar_listas.py

Nao duplica imagem no disco e nao altera o dataset."""
import csv, os

EXP = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(os.path.dirname(EXP), "dataset")
LISTS = os.path.join(EXP, "listas")
os.makedirs(LISTS, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(DS, "mapeamento.csv"), encoding="utf-8"), delimiter=";"))

def paths(split, only_antigo):
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        if only_antigo and r["zip_origem"] == "agro2":
            continue
        out.append(os.path.join(DS, split, "images", r["imagem_nova"]))
    return sorted(out)

sets = {
    "A_train": paths("train", True),   # 206
    "A_valid": paths("valid", True),   # 63
    "A_test":  paths("test",  True),   # 30
    "B_train": paths("train", False),  # 244
    "B_valid": paths("valid", False),  # 74
    "B_test":  paths("test",  False),  # 35
}

for name, ps in sets.items():
    assert ps, name
    for p in ps:
        assert os.path.isfile(p), p
        lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lp = os.path.splitext(lp)[0] + ".txt"
        assert os.path.isfile(lp), lp
    with open(os.path.join(LISTS, name + ".txt"), "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(ps) + "\n")
    print("  %-8s %3d imagens" % (name, len(ps)))

# o conjunto de avaliacao COMUM e o valid/test do dataset antigo:
# nenhum dos dois modelos treina nessas imagens, logo a comparacao e justa
assert set(sets["A_valid"]) <= set(sets["B_valid"])
assert set(sets["A_test"]) <= set(sets["B_test"])
assert not (set(sets["A_valid"]) & set(sets["B_train"])), "valid comum vazou para o train de B!"
assert not (set(sets["A_test"]) & set(sets["B_train"])), "test comum vazou para o train de B!"
assert not (set(sets["A_valid"]) & set(sets["A_train"])), "valid comum vazou para o train de A!"
print("  OK: valid/test comuns nao aparecem no train de nenhum dos dois modelos")


def yaml(fname, train, val, test, comment):
    with open(os.path.join(EXP, fname), "w", newline="\n", encoding="utf-8") as f:
        f.write("# %s\n" % comment)
        f.write("train: %s\n" % os.path.join(LISTS, train).replace("\\", "/"))
        f.write("val: %s\n" % os.path.join(LISTS, val).replace("\\", "/"))
        f.write("test: %s\n\n" % os.path.join(LISTS, test).replace("\\", "/"))
        f.write("nc: 2\nnames:\n  0: erva_daninha\n  1: planta\n")
    print("  -> " + fname)


yaml("A_299.yaml", "A_train.txt", "A_valid.txt", "A_test.txt",
     "MODELO A - dataset antes do Agro-vision2 (299 img: 206/63/30)")
yaml("B_353.yaml", "B_train.txt", "B_valid.txt", "B_test.txt",
     "MODELO B - dataset atual com Agro-vision2 (353 img: 244/74/35)")
# avaliacao comum: train aponta para A_train so para satisfazer o schema; nunca e usado em val()
yaml("eval_comum.yaml", "A_train.txt", "A_valid.txt", "A_test.txt",
     "AVALIACAO COMUM - valid=63 e test=30 retidos de ambos os modelos")
