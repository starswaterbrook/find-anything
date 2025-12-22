from typing import Protocol

from find_anything.models import MaskEmbedding


class MaskSelector(Protocol):
    def select(self, mask_embeddings: list[MaskEmbedding]) -> list[MaskEmbedding]: ...
