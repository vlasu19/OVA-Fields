import torch
import torch.nn.functional as F
import clip
from sentence_transformers import SentenceTransformer
from typing import List


class ClassificationExtractor:
    """
    Extractor class for affordance classification using CLIP and Sentence-BERT models.

    Args:
        clip_model_name (str): Name of the CLIP model to load.
        sentence_model_name (str): Name of the Sentence-BERT model to load.
        aff_class_names (List[str]): List of affordance class names.
        device (str, optional): Device to run the models on. Defaults to "cuda".
        image_weight (float, optional): Weight for image features in classification. Defaults to 1.0.
        affordance_weight (float, optional): Weight for affordance features in classification. Defaults to 1.0.
    """
    AFF_PROMPT = "Here can "
    EMPTY_CLASS = "Other"
    LOGIT_TEMP = 100.0

    def __init__(
        self,
        clip_model_name: str,
        sentence_model_name: str,
        aff_class_names: List[str],
        device: str = "cuda",
        image_weight: float = 1.0,
        affordance_weight: float = 1.0,
    ):
        # Load CLIP and Sentence-BERT models
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_model_name, device=device)

        # Create affordance text strings
        affordance_text_strings = []
        for aff in aff_class_names:
            affordance_text_strings.append(self.AFF_PROMPT + aff)

        # Encode affordance text using Sentence-BERT
        with torch.no_grad():
            all_embedded_affordance_text = sentence_model.encode(
                affordance_text_strings
            )
            all_embedded_affordance_text = (
                torch.from_numpy(all_embedded_affordance_text).float().to(device)
            )

        # Encode affordance text using CLIP
        with torch.no_grad():
            text = clip.tokenize(affordance_text_strings).to(device)
            clip_encoded_text = clip_model.encode_text(text).float().to(device)

        # Delete models after embedding text to free up memory
        del clip_model
        del sentence_model

        # Set class attributes
        self.total_label_classes = len(affordance_text_strings)
        self._clip_embed_size = clip_encoded_text.size(-1)
        self._affordance_embed_size = all_embedded_affordance_text.size(-1)

        # Normalize the text features for both CLIP and affordance models
        self._clip_text_features = F.normalize(clip_encoded_text, p=2, dim=-1)
        self._affordance_features = F.normalize(
            all_embedded_affordance_text, p=2, dim=-1
        )

        # Set weights for image and affordance features
        self._image_weight = image_weight
        self._affordance_weight = affordance_weight

    def calculate_classifications(
        self,
        model_image_features: torch.Tensor,
        model_affordance_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate affordance classifications based on model features.

        Args:
            model_image_features (torch.Tensor): Learned embedding of the image features.
            model_affordance_features (torch.Tensor): Learned embedding of the affordance features.

        Returns:
            torch.Tensor: Weighted sum of the classification probabilities.
        """
        # Ensure input feature sizes match expected dimensions
        assert model_image_features.size(-1) == self._clip_embed_size
        assert model_affordance_features.size(-1) == self._affordance_embed_size

        # Normalize the input features
        model_image_features = F.normalize(model_image_features, p=2, dim=-1)
        model_affordance_features = F.normalize(model_affordance_features, p=2, dim=-1)

        # Calculate similarity logits between image/affordance features and text features
        with torch.no_grad():
            image_logits = model_image_features @ self._clip_text_features.T
            affordance_logits = model_affordance_features @ self._affordance_features.T

        # Ensure logits have correct dimensions
        assert image_logits.size(-1) == self.total_label_classes
        assert affordance_logits.size(-1) == len(self._affordance_features)

        # Calculate weighted sum of softmax probabilities
        return (
            self._image_weight * F.softmax(self.LOGIT_TEMP * image_logits, dim=-1)
            + self._affordance_weight * F.softmax(self.LOGIT_TEMP * affordance_logits, dim=-1)
        ) / (self._image_weight + self._affordance_weight)
