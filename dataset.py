import os
import zipfile
import json
from typing import Dict, Any, List

class DatasetLoader:
    def __init__(self, zip_path: str, extract_dir: str = "data"):
        """
        zip_path: caminho do arquivo .zip (mesma pasta do projeto).
        extract_dir: diretório onde os dados serão extraídos.
        """
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.data_splits: Dict[str, List[Dict[str, Any]]] = {}

    def extract(self) -> None:
        """Extrai o dataset .zip para a pasta especificada"""
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            zf.extractall(self.extract_dir)
        print(f"[OK] Dataset extraído em: {self.extract_dir}")

    def load_jsonl(self, split: str, filename: str) -> None:
        """
        Carrega um arquivo JSONL e armazena no atributo data_splits.
        split: nome do split (train, dev, test)
        filename: nome do arquivo dentro da pasta extraída
        """
        filepath = os.path.join(self.extract_dir, filename)
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        self.data_splits[split] = data
        print(f"[OK] {split} carregado: {len(data)} exemplos")

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        """Retorna os dados do split desejado"""
        if split not in self.data_splits:
            raise ValueError(f"Split {split} não foi carregado ainda.")
        return self.data_splits[split]
