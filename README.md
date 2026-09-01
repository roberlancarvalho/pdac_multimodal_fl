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
   WSI  ─(offline)▶ embeddings de patches ──▶ Ramo B · Histopatologia   ├─▶ Co-atenção
        (Foundation Model de patologia)      (Attention-MIL + Transf.)  │    cross-modal ──▶ risco
                                                                        │    par-a-par        (Cox PH)
   KRAS/TP53/SMAD4/CDKN2A ──▶ Ramo C · Genômico Tabular                 │    (MCAT) + [FUSION]
                              (Embedding por gene)                     ─┘
                                       │
                                       ▼
                         Aprendizado Federado (Flower)
                    servidor agrega pesos · clientes treinam local
```

### Ramo A — Radiômico 3D (`models/branch_a_radiomics.py`)
- **Entrada:** volume de TC pré-processado, `(B, 1, D, H, W)` (ou `(B, 2, …)` para AP+VP).
- **Backbone:** `DenseNet121` 3D da MONAI.
- **Saída:** `(B, embed_dim)` — ou `(B, T, embed_dim)` (`return_tokens=True`), sequência
  de tokens espaciais (`token_grid`, padrão 2×2×2 = 8) para a co-atenção com a histologia.
- Pré-processamento esperado (offline): resample isotrópico, janelamento HU,
  crop/pad em torno da ROI pancreática.

### Ramo B — Histopatologia (`models/branch_b_histology.py`)
- **Não** processa pixels da WSI. Assume `embeddings` de patches gerados
  **offline** por um *foundation model* de patologia (UNI, CONCH, Prov-GigaPath…).
- **Entrada:** `(B, N, patch_feat_dim)` + máscara opcional `(B, N)` para *bags* de tamanho variável.
- **Agregador:** *Attention-Based MIL* (Ilse et al., 2018) + Transformer; `return_tokens=True`
  usa `histology_tokens` *slots* aprendíveis → `(B, K, embed_dim)` tokens histológicos.
- **Saída:** `(B, embed_dim)` (ou `(B, K, embed_dim)`) + pesos de atenção por patch.

### Ramo C — Genômico Tabular (`models/branch_c_genomics.py`)
- **Entrada:** `(B, 4)` `long` — status mutacional de `[KRAS, TP53, SMAD4, CDKN2A]`
  (`0` = wild-type, `1` = mutado, `2` = desconhecido/não sequenciado).
- Um `nn.Embedding` aprendível por `(gene, estado)`.
- **Saída:** `(B, embed_dim)` (via MLP) — ou `(B, 4, embed_dim)` (`return_tokens=True`),
  um token por gene *driver*.
- *TODO:* tipo de variante + frequência alélica (VAF).

### Fusão (`models/fusion_coattention.py` · `fusion_attention.py`)
Dois modos (`fusion_mode` em `configs/default.yaml`):
- **`coattention`** (padrão) — `CrossModalCoAttentionFusion`, estilo **MCAT**
  (Seção 6.1 do artigo): co-atenção **par-a-par e direcional** entre as sequências
  de tokens (Hist→Rad, Rad→Hist, Genômica como *Query* condicionante, etc.),
  `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V`; depois *pooling* por modalidade e leitura
  por token `[FUSION]`. Retorna também `modality_gate` (contribuição por modalidade).
- **`transformer`** (legado) — `CrossModalAttentionFusion`, auto-atenção conjunta
  sobre 1 token por modalidade + `[FUSION]`.
- **Robusto a modalidade ausente** (por amostra): máscara `(B, 3)`.
- **Saída:** `risk (B, n_outputs)`, `fused (B, embed_dim)`.

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
│   ├── fusion_coattention.py   # co-atenção par-a-par (MCAT) — padrão
│   ├── fusion_attention.py     # auto-atenção conjunta — legado
│   └── multimodal_pdac.py
├── federated/
│   ├── server.py
│   ├── client.py
│   ├── engine.py             # laços de treino/avaliação (perda de Cox, C-index)
│   ├── simulation.py
│   ├── reporting.py          # RecordingStrategy + RunRecorder -> outputs/<run>/
│   └── config.py
├── utils/
│   ├── common.py             # seed, device, ponte de parâmetros ↔ Flower
│   └── losses.py             # cox_ph_loss, concordance_index
├── streamlit_app.py          # painel: dispara e acompanha a simulação "em tela"
├── .streamlit/config.toml    # tema (claro/escuro/sistema)
├── outputs/                  # métricas e modelos por execução — NÃO versionado
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

### 4.4 Painel (dashboard Streamlit)

```bash
streamlit run streamlit_app.py     # abre em http://localhost:8501
```

- **Barra lateral:** configura nº de clientes, rodadas, épocas locais, learning
  rate, estratégia (FedAvg/FedProx/FedAdam), amostras sintéticas e dropout de
  modalidade; **Iniciar simulação** dispara `federated/simulation.py` como
  subprocesso.
- **Aba "Treino federado":** auto-atualiza a cada 2 s — status/progresso, KPIs
  (C-index global, perda de Cox treino/avaliação), gráficos por rodada, tabela
  por cliente (instituição) e log do processo.
- **Aba "Atenção — histopatologia":** carrega o modelo global agregado e mostra
  os pesos de atenção do Ramo B (attention-MIL) sobre os patches.
- **Tour guiado:** abre automaticamente na primeira vez (flag `.tour_seen`) e a
  qualquer momento pelo botão **❔ Tour do painel** na barra lateral. Os controles
  e métricas têm *tooltips* (ícone `?`).
- **Tema claro/escuro:** menu ⋮ (canto superior direito) → *Settings* →
  *Appearance* → Light / Dark / System (configurável em `.streamlit/config.toml`).

Cada execução grava em `outputs/<run>/`: `config.json`, `status.json`,
`history.jsonl` (uma linha por rodada), `global_model.pt`, `tb/` (TensorBoard) e
`c_index.png` (ao final). O seletor **Execução** no topo do painel lista todas as
execuções para reabrir/comparar. `RecordingStrategy` (`federated/reporting.py`)
embrulha a estratégia do Flower para registrar as métricas sem alterar a agregação.

### 4.5 TensorBoard

```bash
tensorboard --logdir outputs        # abre em http://localhost:6006
```

Registrado por execução em `outputs/<run>/tb/`:

| Tipo | Tags |
|------|------|
| Scalars (por rodada) | `global/c_index`, `global/loss_eval`, `global/loss_train`, `clients/<Cliente N>/…` |
| Scalars (atenção) | `attention_branch_b/entropy_norm` — entropia normalizada da atenção do Ramo B (1.0 = uniforme, ~0 = concentrada) |
| Histogramas (por rodada) | `weights/branch_a`, `weights/branch_b`, `weights/branch_c`, `weights/fusion`, `attention_branch_b/weights` |
| Scalars locais (por cliente) | `local/train_loss` (por época), `local/eval_loss`, `local/c_index` — em `tb/local_cliente_<n>/` |

O gráfico **`c_index.png`** (C-index + perdas de Cox × rodada) é gerado ao final
de cada execução e aparece no painel em *Exportações*.

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

Alinhamento com a Seção 6.1 do artigo:

- [x] Fusão por **co-atenção cross-modal par-a-par (MCAT)** — `fusion_coattention.py`.
- [x] Ramo A emite **tokens espaciais**; Ramo B emite **K tokens** histológicos;
  Ramo C emite **1 token por gene** *driver*.
- [ ] Ramo A: fases **AP + VP** com encoder compartilhado (hoje: `ct_in_channels=2`).
- [ ] Ramo C: **tipo de variante + VAF** além do status mutacional.
- [ ] Regularização de **balanceamento entre modalidades** no treino.
- [ ] Explicabilidade: **SHAP** (Ramo C + clínico) e **Grad-CAM 3D** (Ramo A).
- [ ] Cabeças **multitarefa**: diagnóstico + subtipagem molecular + prognóstico.

Geral:

- [ ] Implementar `MultimodalPDACDataset` para o(s) dataset(s) reais.
- [ ] Pipeline de pré-processamento de TC (MONAI transforms) e extração de patches/embeddings de WSI.
- [ ] Avaliação centralizada no servidor com coorte de validação externa.
- [ ] Privacidade: *secure aggregation* / DP-SGD (`flwr` + Opacus).
- [ ] Estratégias para não-IID entre instituições (FedProx / FedBN).
- [ ] Testes (`pytest`) para shapes dos ramos, máscara de modalidade e `cox_ph_loss`.

---

## 8. Licença

Código sob licença **MIT** — ver [`LICENSE`](LICENSE).

Os **dados de pacientes não são cobertos** por esta licença e permanecem sob a
LGPD e sob os termos dos comitês de ética e dos acordos de uso de dados de cada
instituição participante.
