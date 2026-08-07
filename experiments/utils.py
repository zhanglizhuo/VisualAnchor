"""
Shared utilities for SCB5 experiment scripts.

Usage:
    from utils import CLASS_NAME_MAP, read_label, load_clip_model
"""

CLASS_NAME_MAP = {
    "BowHead": "bowing head",
    "TurnHead": "turning head",
    "hand-raising": "raising hand",
    "On-stage interaction": "on stage interaction",
    "blackboard-writing": "writing on blackboard",
    "blackBoard": "standing near a blackboard or whiteboard",
    "stand": "standing at the front of a classroom",
    "guide": "guiding students in classroom",
    "answer": "answering questions in class",
}


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return int(line.split()[0])


def match_class_name(response: str, true_class: str, all_classes) -> bool:
    """Match MLLM response against true class, disambiguating substring conflicts.

    The naive ``true_class.lower() in response.lower()`` fails when one class
    name is a substring of another (e.g. ``blackBoard`` vs ``blackboard-writing``).
    This function collects every valid class that appears in the response and
    picks the *longest* match — i.e. the most specific class name.

    Args:
        response: Raw MLLM output.
        true_class: Ground-truth class name.
        all_classes: All valid class names for this dataset.

    Returns:
        True if the most specific class name found in the response
        matches *true_class*.
    """
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    true_lower = true_class.lower()

    if resp == true_lower:
        return True

    valid = [c.lower() for c in all_classes]
    matched = [c for c in valid if c in resp]

    if len(matched) == 1:
        return matched[0] == true_lower
    elif len(matched) > 1:
        return max(matched, key=len) == true_lower
    else:
        return False


def load_clip_model(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k"):
    """Load a CLIP model with its native preprocessing.

    Uses ``open_clip.create_model_and_transforms`` which returns the
    correct image normalization for each model architecture.  This
    ensures reproducibility without hard-coding a single normalization
    for all backbones (which would break SigLIP, EVA-02, etc.).
    """
    import open_clip

    model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, transform
