import hydra
from omegaconf import DictConfig

from src.models import BaseModel, RagModel

import warnings


@hydra.main(version_base=None, config_path="configs/base", config_name="parameters")
def main(cfg: DictConfig) -> None:
    base_model = BaseModel(model_name=cfg.get("model_name"))
    rag_model = RagModel(
        model_name=cfg.get("model_name"),
        chunk_size=cfg.get("rag_chunk_size"),
        chunk_overlap=cfg.get("rag_chunk_overlap")
    )
    question = "What songs is Taylor Swift known for?"
    print(f"Question: {question}")
    print(f"Base answer: {base_model.chat(question)}")
    print(f"Answer with RAG: {rag_model.chat(question)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
