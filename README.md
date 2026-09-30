# Agro-vision — dataset unificado e treinos YOLO11n

Detecção de `erva_daninha` e `planta` com YOLO11n. Este branch contém o dataset
unificado a partir de 4 exports separados e três treinos comparativos.

## Dataset

`dataset/` — **353 imagens únicas**, 4977 anotações, formato YOLO.

| split | imagens | erva_daninha | planta | labels vazios |
|-------|--------:|-------------:|-------:|--------------:|
| train | 244 | 892 | 2846 | 8 |
| valid | 74  | 221 | 613  | 2 |
| test  | 35  | 98  | 307  | 1 |

Unificado a partir de `Agro-vision-game.v1i.yolov11.zip`, `Agro-vision.v1i.yolov11.zip`,
`train.zip` e `Agro-vision2.v1i.yolov11.zip`.

O que foi corrigido na unificação:

- **84 duplicatas exatas removidas.** Os 3 primeiros exports somavam 383 arquivos
  rotulados, mas apenas 299 imagens distintas — o hash `.rf.<md5>` que o Roboflow
  põe no nome mascarava as repetições. A deduplicação foi por SHA256 do conteúdo.
- **Vazamento entre splits eliminado.** 13 imagens estavam em splits conflitantes
  entre os exports (ex.: uma mesma foto em `test` num export e `train` em outro).
  Os splits foram refeitos do zero com `seed=42`, agrupando variantes da mesma foto
  (`X`, `X-2-`, `X-3-`) para que nunca caiam em lados opostos.
- **Labels divergentes resolvidos.** 23 grupos de imagens idênticas tinham `.txt`
  diferentes; a divergência era só arredondamento na 4ª casa decimal (requantização
  entre exports do Roboflow). Manteve-se o export mais recente como canônico.

Arquivos de apoio:

- `dataset/data.yaml` — config do dataset. **O campo `path:` é absoluto**; ajuste após clonar.
- `dataset/mapeamento.csv` — rastreabilidade: nome novo → ZIP de origem, nome original,
  SHA256 e quais duplicatas foram descartadas.
- `dataset/_nao_anotadas/` — 84 imagens cruas sem label, **fora** do `data.yaml`.
  82 já existem anotadas nos exports; 2 nunca foram anotadas.

## Treinos

Três treinos com `yolo11n.pt` como base, em GTX 1650 (4 GB). AMP foi desabilitado
automaticamente pela ultralytics (a série GTX 16xx falha no teste de fp16), então
rodaram em fp32.

| modelo | train | patience | epochs | tempo |
|--------|------:|---------:|-------:|------:|
| `A_299` | 206 img (sem Agro-vision2) | 20 | 200 (melhor: 190) | 26,7 min |
| `B_353` | 244 img | 20 | 106 (early stop, melhor: 86) | 16,3 min |
| `C_353_p60` | 244 img | 60 | 200 | 30,9 min |

Hiperparâmetros comuns: `batch=8 imgsz=640 workers=2 cache=False seed=0 device=0`.

### Resultados

Média ponderada sobre as 109 imagens retidas (63+30 de cenas antigas, 11+5 da
rajada nova de 25/09). Nenhum modelo treinou nessas imagens.

| modelo | mAP50 | mAP50-95 | precisão | recall |
|--------|------:|---------:|---------:|-------:|
| A_299 | 0,9802 | 0,8631 | 0,9633 | 0,9608 |
| B_353 | 0,9811 | 0,8583 | 0,9721 | **0,9616** |
| **C_353_p60** | **0,9830** | **0,8710** | **0,9821** | 0,9599 |

**`C_353_p60` é o melhor modelo** — pesos em `runs_treino/C_353_p60/weights/best.pt`.

Dois achados por trás desse resultado:

1. **`patience=20` era curto demais.** O B parou no epoch 86 enquanto sua média
   móvel de mAP50-95 ainda subia (ep74=0,8513 → ep90=0,8562 → ep106=0,8597). Com
   só 74 imagens de validação a métrica oscila o suficiente para passar 20 epochs
   sem novo recorde absoluto mesmo em tendência de alta. Com `patience=60`, o mesmo
   dataset sobe de 0,8583 para 0,8710 de mAP50-95.
2. **O dado novo rende onde ele cobre.** Nas cenas de 25/09 que o A nunca viu, o C
   ganha +8,9% de mAP50-95 no valid e +12,8% no test (recall +10,6%). Nas cenas
   antigas não há ganho porque o mAP50 já está em 0,98 — o conjunto está saturado.

### ⚠️ As métricas absolutas estão infladas

O dataset são ~28 rajadas de captura de webcam/screenshot. Frames da mesma rajada
estão a segundos de distância e são visualmente quase idênticos:

| janela | imagens de valid com um frame de treino a essa distância |
|--------|--------------------------------------------------------:|
| ±2 s  | 36/74 (49%) |
| ±5 s  | 47/74 (64%) |
| ±30 s | 69/74 (93%) |

Como o split é por imagem e não por rajada, **mAP50 ≈ 0,98 reflete quase-memorização
de cena, não generalização**. Não espere esse número num campo novo. Para medir a
generalização real, o split precisa ser por rajada de captura.

Outro ponto: em `conf=0,25` fixo, o A ainda tem a melhor precisão nas cenas antigas
(0,9269 vs 0,9128 do C; 45 vs 54 falsos positivos). O `val()` da ultralytics mede no
limiar ótimo, não num ponto de operação fixo. Se for usar o C em produção, suba o
`conf` para ~0,35.

## Como reproduzir

```bash
python -m venv .venv                  # Python 3.11 (o PyTorch ainda não suporta 3.14)
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/pip install ultralytics

python experimentos/gerar_listas.py   # regenera as listas com os caminhos desta máquina
python experimentos/treinar.py        # treinos A e B
python experimentos/treinar_C.py      # treino C (patience=60)
python experimentos/final.py          # avaliação dos 3 nos mesmos conjuntos retidos
```

`experimentos/listas/*.txt` guardam caminhos absolutos — rode `gerar_listas.py`
antes de qualquer treino após clonar.

## Estrutura

```
dataset/                  353 imagens + labels, data.yaml, mapeamento.csv
  train|valid|test/       images/ e labels/
  _nao_anotadas/          84 imagens cruas, fora do data.yaml
experimentos/
  gerar_listas.py         regenera listas e YAMLs (rode após clonar)
  treinar.py              treinos A e B
  treinar_C.py            treino C, patience=60
  comparar.py             A vs B nos conjuntos comuns
  final.py                os 3 modelos, tabela final
  analise_valid.py        TP/FP/FN imagem a imagem em conf=0,25
  eval_cenas_novas.py     avaliação só nas cenas de 25/09
runs_treino/              pesos, curvas e matrizes de confusão dos 3 treinos
```

O `.venv/` (5 GB) está no `.gitignore`.
