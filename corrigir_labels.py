import os

pastas = [
    "dataset/labels/train",
    "dataset/labels/val"
]

for pasta in pastas:

    arquivos = os.listdir(pasta)

    for arquivo in arquivos:

        if arquivo.endswith(".txt"):

            caminho = os.path.join(pasta, arquivo)

            with open(caminho, "r") as f:
                linhas = f.readlines()

            novas_linhas = []

            for linha in linhas:

                dados = linha.split()

                if dados[0] == "15":
                    dados[0] = "0"

                elif dados[0] == "16":
                    dados[0] = "1"

                novas_linhas.append(" ".join(dados))

            with open(caminho, "w") as f:
                f.write("\n".join(novas_linhas))

print("Correção concluída!")