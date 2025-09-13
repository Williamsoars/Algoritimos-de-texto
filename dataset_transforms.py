# my_sener_lib/dataset_transforms.py
import json
from typing import Dict, Any

def scholar_xl_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforma amostras do formato Scholar-XL para formato padrão.
    Adapte conforme a estrutura real dos seus arquivos.
    """
    # Caso 1: Formato já correto
    if "text" in sample and "entities" in sample:
        return sample
    
    # Caso 2: Formato com 'content' e 'annotations'
    if "content" in sample and "annotations" in sample:
        return {
            "text": sample["content"],
            "entities": [
                {
                    "start": ann["start"],
                    "end": ann["end"], 
                    "label": ann["label"]
                }
                for ann in sample["annotations"]
            ]
        }
    
    # Caso 3: Formato com outros nomes de campos
    # Adicione mais casos conforme necessário
    
    # Se não reconhecer, retorna como está (pode causar erro depois)
    print(f"⚠️  Formato desconhecido: {list(sample.keys())}")
    return sample

def create_scholar_xl_dataset(file_path: str, tokenizer_name: str, label2id_path: str, **kwargs):
    """Factory function para criar dataset Scholar-XL."""
    with open(label2id_path, 'r') as f:
        label2id = json.load(f)
    
    return BaseNERDataset(
        file_path=file_path,
        tokenizer_name=tokenizer_name,
        label2id=label2id,
        transform=scholar_xl_transform,
        **kwargs
    )
