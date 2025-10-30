import torch
from transformers import CLIPProcessor, CLIPModel
from utils.logger import setup_logger
from config import CLIP_MODEL_NAME, CLIP_RELEVANCE_THRESHOLD, DEVICE

logger = setup_logger(__name__)

_model = None
_processor = None

def _ensure_clip_loaded():
    global _model, _processor
    if _model is None or _processor is None:
        logger.info(f"[semantic_filter] Loading CLIP model: {CLIP_MODEL_NAME} on {DEVICE}")
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _model.eval()

@torch.no_grad()
def clip_text_relevance(query: str, caption: str, threshold: float = CLIP_RELEVANCE_THRESHOLD, return_score=False):
    """Compute semantic similarity between query and image caption using CLIP text encoder."""
    _ensure_clip_loaded()

    if not caption:
        return (True, 0.0) if return_score else True  # allow uncaptioned images

    inputs = _processor(text=[query, caption], return_tensors="pt", padding=True).to(DEVICE)
    text_embeds = _model.get_text_features(**inputs)

    q_emb, c_emb = text_embeds[0], text_embeds[1]
    score = torch.cosine_similarity(q_emb, c_emb, dim=0).item()

    logger.debug(f"[semantic_filter] CLIP text sim ({query[:40]} vs {caption[:40]}) = {score:.3f}")

    if return_score:
        return (score >= threshold, score)
    return score >= threshold
