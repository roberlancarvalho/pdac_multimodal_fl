# Guia de dados — o que fazer para sair do sintético

O código já implementa **todo o pipeline**. O que falta é o que **só você / o
consórcio** pode fazer: conseguir dados reais, pré-processá-los, montar o
manifesto e (para a federação de verdade) tratar de ética e infraestrutura.

Este guia é o passo a passo.

---

## 0. Ordem recomendada

1. **Prova de conceito com dado público** (1 instituição, sem federação) → você valida
   o modelo de verdade.
2. **Federação simulada com 2–3 partições** do dataset público → valida FedAvg/DP/FedBN.
3. **Federação real** entre instituições → precisa de ética + rede (Seção 5).

---

## 1. Datasets públicos que servem

| Dataset | Modalidades | Onde | Observações |
|---|---|---|---|
| **CPTAC‑PDA** | TC + WSI (H&E) + proteômica/genômica + clínico | TCIA (imagens) + PDC (ômicas) | melhor opção multimodal para PDAC |
| **TCGA‑PAAD** | WSI + RNA‑seq + mutações (MAF) + clínico + (parte com TC) | GDC Portal / cBioPortal | 185 casos; mutações prontas em MAF |
| **MSD Task07 (Pancreas)** | TC + segmentação | medicaldecathlon.com | só imagem+máscara, bom para o Ramo A |
| **PANORAMA / PANDA (se liberado)** | TC | grand‑challenge.org | grandes coortes de detecção |

Para o **mestrado**, o caminho mais rápido: **TCGA‑PAAD** (genômica + WSI + clínico
prontos) + **MSD Task07** ou o subconjunto de TC do CPTAC‑PDA para o Ramo A.

---

## 2. Pré‑processar cada modalidade (offline)

### 2.1 TC (Ramo A)

Objetivo: gerar `.nii.gz` por fase (AP e VP) — ou 1 volume — já no grid certo.

1. **DICOM → NIfTI**: `dcm2niix -o saida/ -f %p_%s pasta_dicom/`
   (ou `SimpleITK` / `dicom2nifti`).
2. O resto (**resample isotrópico, janelamento HU pancreático, crop da ROI,
   resize**) o `data.preprocessing.ct_transforms()` já faz **na hora**, dentro do
   `Dataset`. Você só precisa do NIfTI cru + a orientação correta.
3. *(Opcional, recomendado)* Segmentar o pâncreas com **nnU‑Net** (modelo do MSD
   Task07) e salvar a máscara — depois troque `CropForegroundd(source_key="image")`
   por um crop guiado pela máscara em `ct_transforms`.

Se você só tem **1 fase**: use a coluna `ct_path` no manifesto e ponha
`model.radiomics_phases: 1`, `model.ct_in_channels: 1`.

### 2.2 WSI / histopatologia (Ramo B)

O modelo **não** lê o `.svs`. Ele lê um `.npy` `(N, F)` de embeddings de patches.
Gere‑o **offline**:

```bash
# precisa de: pip install openslide-python timm  + binário OpenSlide + acesso ao UNI no HuggingFace
export HF_TOKEN=hf_xxx
python -m data.wsi_patching --slide LAMINA.svs --out wsi_emb/LAMINA.npy --encoder uni --patch 256 --mpp 0.5
```

`data/wsi_patching.py` é um **scaffold**: detecção de tecido simples + encoder
plugável (`uni`, `virchow`, `conch`, ou `random` para testar). Para uso sério,
troque a detecção de tecido pelo segmentador do **CLAM** ou **Trident**
(pipelines maduros de *patching*), mantendo a saída `.npy (N, F)`.

`F` (dimensão do embedding) deve casar com `model.patch_feat_dim`
(UNI = 1024, Prov‑GigaPath = 1536, Virchow = 2560).

### 2.3 Genômica (Ramo C)

- **TCGA‑PAAD**: baixe o **MAF** do GDC (ou `Mutations` do cBioPortal). Um arquivo
  por paciente, ou um MAF grande — filtre por `Tumor_Sample_Barcode`.
- `data.preprocessing.parse_mutations` já entende MAF (`Hugo_Symbol`,
  `Variant_Classification`, `t_alt_count`/`t_depth` → VAF) e um CSV simples
  (`gene,status,variant_type,vaf`).
- No manifesto: coluna `mutations` = caminho do MAF/CSV **ou** colunas diretas
  `KRAS_status,KRAS_vtype,KRAS_vaf,TP53_status,...`.

### 2.4 Clínico (Ramo D)

- Reúna as variáveis (ex.: idade, **CA 19‑9**, IMC, tamanho do tumor, albumina;
  sexo, estágio AJCC, ECOG PS, localização, ressecabilidade).
- **Padronize as contínuas** (z‑score) e **codifique as categóricas** como
  inteiros `0..n‑1` — *offline*, antes do manifesto.
- No `configs/default.yaml`:
  ```yaml
  data:
    clinical_continuous_cols: [age, ca19_9, bmi, tumor_size_mm, albumin]
    clinical_categorical_cols: [sex, ajcc_stage, ecog_ps, tumor_location]
  model:
    clinical_n_continuous: 5                 # = len(clinical_continuous_cols)
    clinical_cat_cardinalities: [2, 4, 5, 3] # nº de categorias de cada categórica
  ```

### 2.5 Rótulos

- `time` (meses até óbito/último follow‑up), `event` (1 = óbito, 0 = censura) — para o Cox.
- `dx` (1 = PDAC, 0 = não‑PDAC/benigno) — se você tiver controles.
- `subtype` (0 = classical, 1 = basal‑like) — do Moffitt/Collisson, se disponível;
  vazio se não tiver.

---

## 3. Montar o manifesto

Um CSV por instituição (`data/manifest_template.csv` é o exemplo):

```csv
patient_id,ct_ap,ct_vp,wsi_emb,mutations,age,ca19_9,bmi,tumor_size_mm,albumin,sex,ajcc_stage,ecog_ps,tumor_location,time,event,dx,subtype,split
TCGA-XX-0001,,,wsi_emb/TCGA-XX-0001.npy,mut/TCGA-XX-0001.maf,-0.3,1.8,0.1,-0.5,0.2,1,3,1,0,14.2,1,1,1,train
```

- Caminhos **relativos a `data.data_root`**.
- Célula vazia = modalidade/rótulo ausente (o modelo lida com isso via máscara).
- Faça o **split** `train`/`val` por paciente (nunca vaze paciente entre eles).
- Aponte `federated.central_eval.manifest` para o CSV da coorte **held‑out** do
  servidor (idealmente de **outra** instituição = validação externa).

Aponte no config e rode:
```yaml
data: { manifest_csv: "/caminho/manifest.csv", data_root: "/caminho/dados" }
```
```bash
python -m federated.simulation --config seu.yaml --num-clients 3 --num-rounds 20
```

---

## 4. Ajustes de modelo prováveis

| Você tem | Ajuste em `configs/default.yaml` |
|---|---|
| 1 fase de TC | `radiomics_phases: 1`, `ct_in_channels: 1` |
| embeddings UNI (F=1024) | `patch_feat_dim: 1024` (padrão) |
| sem histopatologia | remova a coluna `wsi_emb` — a fusão ignora |
| sem subtipo molecular | `w_subtype: 0` (ou deixe a coluna vazia) |
| poucos pacientes / backbone pré‑treinado | `freeze_radiomics: true` |
| dados muito heterogêneos entre centros | `fedbn: true`, `strategy: FedProx` |
| exigência de privacidade formal | `dp.enabled: true` e reporte o **ε** (aparece no painel/`config.json`) |

---

## 5. Federação REAL entre instituições (o que exige processo, não código)

1. **Ética e dados**
   - Aprovação no **CEP/CONEP** (Plataforma Brasil) de cada instituição.
   - **Acordo de uso de dados** (DUA) entre os centros e a UFRJ.
   - **DPIA / RIPD** (LGPD, art. 38) documentando o fluxo federado (dados não saem;
     só pesos + ruído DP saem).
2. **Infraestrutura**
   - 1 servidor (pode ser na UFRJ) alcançável pelos clientes; abra a porta do
     `server_address` (padrão `8080`) com **TLS**.
   - Cada instituição roda `python -m federated.client --cid <i> --num-clients <N>
     --server-address <host:porta> --config <local>.yaml` na própria rede, com o
     próprio `manifest_csv`.
   - Combine hardware mínimo (GPU ajuda muito no Ramo A).
3. **Protocolo do experimento**
   - Fixe seeds, versão do código (tag git), config idêntico entre centros.
   - Defina a coorte de **validação externa** (1 centro que só avalia).
   - Rode também os **baselines**: cada modalidade isolada, `late fusion`, e
     comparação com estadiamento AJCC (C‑index 0,54–0,57 do artigo).
4. **Reprodutibilidade / publicação**
   - Guarde `outputs/<run>/` de cada rodada (config, history, TensorBoard, modelo).
   - Reporte: C‑index (distribuído **e** central), AUC diagnóstico, IC95%,
     calibração, e o **ε** do DP.

---

## 6. O que ainda é trabalho de pesquisa (não bloqueante)

- **Secure aggregation** (SecAgg+) — hoje o servidor vê os pesos individuais dos
  clientes antes de agregar. Exige migrar para a API `flwr run` / `ServerApp` do
  Flower ≥ 1.13 e ativar o `SecAggPlusWorkflow`.
- **Não‑IID avançado** — FedBN já está; avaliar FedProx/FedAdam/scaffold nos seus dados.
- **Segmentação do tumor** como pré‑etapa (nnU‑Net) para o crop guiado do Ramo A.
- **Curvas de sobrevida** (Kaplan‑Meier por grupo de risco) com `lifelines`
  para a discussão clínica.
