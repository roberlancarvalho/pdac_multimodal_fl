# Pipeline Multimodal Federado para PDAC

Implementação prática (mestrado — PPGI/UFRJ) de um modelo **multimodal** para
predição de **risco / sobrevida** em **adenocarcinoma ductal de pâncreas (PDAC)**,
treinado de forma **federada** (os dados de cada instituição nunca saem da
instituição).

> ⚠️ **Software as a Medical Device (SaMD) — pesquisa.** Este repositório é
> artefato de pesquisa acadêmica. Não é dispositivo médico aprovado e não deve
> ser usado para decisão clínica. Dados de pacientes estão sujeitos à LGPD e aos
> comitês de ética das instituições participantes.

---

## 1. Arquitetura

Três ramos independentes extraem uma representação (`embedding` de dimensão
`embed_dim`) de cada modalidade. Um módulo de **Atenção Cruzada (Cross-Modal
Attention)** funde os três `embeddings` e uma cabeça de risco produz o
log-hazard usado na perda de Cox.

```
   TC 3D (NIfTI) ──▶ Ramo A · Radiômico 3D      (MONAI DenseNet121 3D)  ─┐
                                                                        │
   WSI  ─(offline)▶ embeddings de patches ──▶ Ramo B · Histopatologia   ├─▶ Cross-Modal
        (Foundation Model de patologia)      (Attention-MIL + Transf.)  │    Attention ──▶ risco
                                                                        │   (Transformer      (Cox PH)
   KRAS/TP53/SMAD4/CDKN2A ──▶ Ramo C · Genômico Tabular                 │    + token [FUSION])
                              (Embedding por gene + MLP)               ─┘
                                       │
                                       ▼
                         Aprendizado Federado (Flower)
                    servidor agrega pesos · clientes treinam local
```

### Ramo A — Radiômico 3D (`models/branch_a_radiomics.py`)
- **Entrada:** volume de TC pré-processado, tensor `(B, 1, D, H, W)`.
- **Backbone:** `DenseNet121` 3D da MONAI; cabeça substituída por projeção para `embed_dim`.
- **Saída:** `(B, embed_dim)`.
- Pré-processamento esperado (offline): resample isotrópico, janelamento HU,
  crop/pad em torno da ROI pancreática.

### Ramo B — Histopatologia (`models/branch_b_histology.py`)
- **Não** processa pixels da WSI. Assume `embeddings` de patches gerados
  **offline** por um *foundation model* de patologia (UNI, CONCH, Prov-GigaPath…).
- **Entrada:** `(B, N, patch_feat_dim)` + máscara opcional `(B, N)` para *bags* de tamanho variável.
- **Agregador:** *Attention-Based MIL* (Ilse et al., 2018) com bloco Transformer opcional.
- **Saída:** `(B, embed_dim)` + pesos de atenção por patch (interpretabilidade).

### Ramo C — Genômico Tabular (`models/branch_c_genomics.py`)
- **Entrada:** `(B, 4)` `long` — status mutacional de `[KRAS, TP53, SMAD4, CDKN2A]`
  (`0` = wild-type, `1` = mutado, `2` = desconhecido/não sequenciado).
- Um `nn.Embedding` aprendível por `(gene, estado)` + MLP.
- **Saída:** `(B, embed_dim)`.

### Fusão — Cross-Modal Attention (`models/fusion_attention.py`)
- Adiciona um *modality token* a cada `embedding`, concatena um token `[FUSION]`
  (tipo CLS) e passa a sequência de 4 tokens por um Transformer.
- **Robusto a modalidade ausente:** máscara `(B, 3)` — pacientes sem uma das
  modalidades continuam sendo processados.
- **Saída:** `risk (B, n_outputs)` e `fused (B, embed_dim)`.

### Modelo completo (`models/multimodal_pdac.py`)
`MultimodalPDACModel` orquestra os quatro módulos. É esta `nn.Module` cujos
pesos o Flower serializa e agrega. Cada ramo pode ser **congelado**
individualmente (`freeze_radiomics`, `freeze_histology`, `freeze_genomics`) para
treinar apenas a fusão.

### Aprendizado Federado (`federated/`)
- `server.py` — `start_server` + estratégia (`FedAvg` / `FedProx` / `FedAdam`),
  agregação de C-index ponderada por nº de amostras. **Nenhum dado de paciente
  passa pelo servidor.**
- `client.py` — `NumPyClient`: recebe pesos globais → treina local
  (`local_epochs`) → devolve pesos + métricas.
- `engine.py` — laços de treino/avaliação (perda de Cox, C-index).
- `simulation.py` — roda servidor + N clientes virtuais em um único processo.
- `config.py` — carrega `configs/default.yaml`.

---

## 2. Estrutura do repositório

```
pdac_multimodal_fl/
├── configs/
│   └── default.yaml          # hiperparâmetros de modelo / treino / federação
├── data/
│   ├── dataset.py            # MultimodalPDACDataset (real, a implementar) + SyntheticPDACDataset
│   ├── raw/                  # dados brutos por instituição — NÃO versionado
│   └── processed/            # dados pré-processados — NÃO versionado
├── models/
│   ├── branch_a_radiomics.py
│   ├── branch_b_histology.py
│   ├── branch_c_genomics.py
│   ├── fusion_attention.py
│   └── multimodal_pdac.py
├── federated/
│   ├── server.py
│   ├── client.py
│   ├── engine.py
│   ├── simulation.py
│   └── config.py
├── utils/
│   ├── common.py             # seed, device, ponte de parâmetros ↔ Flower
│   └── losses.py             # cox_ph_loss, concordance_index
├── requirements.txt
└── README.md
```

---

## 3. Instalação

Requer **Python ≥ 3.10** (desenvolvido em 3.13).

```powershell
# Windows / PowerShell
.\venv\Scripts\Activate.ps1

# Instale o PyTorch adequado à sua GPU antes do restante:
#   https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

```bash
# Linux / macOS
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Como executar localmente

### 4.1 Smoke test do modelo (sem dados)

Cada arquivo de modelo tem um bloco `__main__` com tensores fictícios:

```bash
python -m models.multimodal_pdac
python -m models.fusion_attention
```

### 4.2 Simulação federada (1 processo, recomendado para desenvolver)

Usa `SyntheticPDACDataset` particionado entre os clientes:

```bash
python -m federated.simulation --num-clients 3 --num-rounds 5
```

### 4.3 Federação "real" (servidor + clientes em terminais separados)

```bash
# Terminal 1 — servidor
python -m federated.server --num-rounds 10

# Terminal 2 — cliente 0
python -m federated.client --cid 0 --num-clients 2

# Terminal 3 — cliente 1
python -m federated.client --cid 1 --num-clients 2
```

Para federação entre máquinas, ajuste `federated.server_address` em
`configs/default.yaml` (ou passe `--server-address host:porta`) e garanta
conectividade/TLS entre os nós.

---

## 5. Ligando os seus dados

1. Pré-processe **offline**, em cada instituição:
   - TC → NIfTI resampleado + crop da ROI pancreática (`data/processed/ct/`);
   - WSI → patches → `embeddings` do *foundation model* (`data/processed/wsi_emb/`);
   - painel genético → colunas `KRAS, TP53, SMAD4, CDKN2A` ∈ {0,1,2}.
2. Monte um **manifesto CSV** por instituição (1 linha por paciente) com os
   caminhos e os rótulos de sobrevida (`time`, `event`).
3. Implemente o carregamento em `MultimodalPDACDataset.__getitem__`
   (`data/dataset.py`) retornando o `dict` documentado no topo do arquivo.
4. Aponte `data.manifest_csv` / `data.data_root` no `configs/default.yaml` e
   troque o `build_dataloaders` em `federated/client.py`.

---

## 6. Convenções

- `embed_dim` é **compartilhado** pelos três ramos e pela fusão — altere em um
  único lugar (`configs/default.yaml`).
- A ordem canônica das modalidades é `("radiomics", "histology", "genomics")`
  (ver `models/fusion_attention.MODALITIES`).
- A ordem canônica dos genes é `("KRAS", "TP53", "SMAD4", "CDKN2A")`
  (ver `models/branch_c_genomics.PDAC_DRIVER_GENES`).
- Nada em `data/raw/` ou `data/processed/` é versionado (`.gitignore`).

---

## 7. Roadmap

- [ ] Implementar `MultimodalPDACDataset` para o(s) dataset(s) reais.
- [ ] Pipeline de pré-processamento de TC (MONAI transforms) e extração de patches/embeddings de WSI.
- [ ] Avaliação centralizada no servidor com coorte de validação externa.
- [ ] Privacidade: *secure aggregation* / DP-SGD (`flwr` + Opacus).
- [ ] Estratégias para não-IID entre instituições (FedProx / FedBN).
- [ ] Testes (`pytest`) para shapes dos ramos, máscara de modalidade e `cox_ph_loss`.
```
