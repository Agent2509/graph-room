"""FastEmbed ONNX embeddings on CPU. No API key, no rate limit."""

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Wraps FastEmbed. Use the instance as a callable: ``embed(["a", "b"])``."""

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            self._model = TextEmbedding(model_name=model_name, local_files_only=True)
        except Exception:  # not cached yet; download once
            self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def one(self, text: str) -> list[float]:
        return self([text])[0]
