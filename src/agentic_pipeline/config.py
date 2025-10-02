import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    project_root: Path
    data_dir: Path
    kb_dir: Path
    prices_csv: Path
    results_dir: Path

    openai_api_key: Optional[str]
    llm_model: str
    embedding_model: str
    prompt_version: str

    @staticmethod
    def from_env() -> "Config":
        load_dotenv(override=False)
        # Resolve project root from this file location
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data"
        kb_dir = data_dir / "kb"
        prices_csv = data_dir / "prices.csv"
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        prompt_version = os.environ.get("PROMPT_VERSION", "v1")

        return Config(
            project_root=project_root,
            data_dir=data_dir,
            kb_dir=kb_dir,
            prices_csv=prices_csv,
            results_dir=results_dir,
            openai_api_key=openai_api_key,
            llm_model=llm_model,
            embedding_model=embedding_model,
            prompt_version=prompt_version,
        )


