import torch

from models.multimodal_pdac import MultimodalPDACModel
from utils.xai import clinical_shap, genomics_shap, radiomics_gradcam


def test_gradcam_shape_and_range(model_cfg, batch):
    model = MultimodalPDACModel(**model_cfg)
    cam = radiomics_gradcam(model, {"ct_volume": batch["ct_volume"], "mutation_status": batch["mutation_status"]})
    assert cam.shape == batch["ct_volume"].shape[:1] + batch["ct_volume"].shape[-3:]
    assert torch.isfinite(cam).all()
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0 + 1e-5


def test_genomics_shap_additivity(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    res = genomics_shap(model, torch.tensor([[1, 1, 0, 1]]), nsamples=32)
    assert res["genes"] == ["KRAS", "TP53", "SMAD4", "CDKN2A"]
    assert abs(res["base_value"] + sum(res["shap"]) - res["prediction"]) < 1e-3


def test_clinical_shap_additivity(model_cfg):
    model = MultimodalPDACModel(**model_cfg)
    num = torch.randn(1, model_cfg["clinical_n_continuous"])
    cat = torch.zeros(1, len(model_cfg["clinical_cat_cardinalities"]), dtype=torch.long)
    res = clinical_shap(model, num, cat, nsamples=32)
    assert len(res["fields"]) == num.shape[1] + cat.shape[1]
    assert abs(res["base_value"] + sum(res["shap"]) - res["prediction"]) < 1e-3
