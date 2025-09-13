# scripts/inspect_dataset.py
import json
from my_sener_lib.dataset import BaseNERDataset
from my_sener_lib.dataset_transforms import scholar_xl_transform

def inspect_scholar_xl_file(file_path: str):
    """Inspeciona a estrutura do arquivo Scholar-XL."""
    print(f"🔍 Inspecionando {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        first_lines = []
        for i, line in enumerate(f):
            if i >= 5:  # Primeiras 5 linhas
                break
            try:
                data = json.loads(line)
                first_lines.append(data)
            except:
                print(f"❌ Erro na linha {i+1}")
                break
    
    print("📋 Primeiras amostras:")
    for i, sample in enumerate(first_lines):
        print(f"\nAmostra {i+1}:")
        print(f"  Keys: {list(sample.keys())}")
        if 'text' in sample:
            print(f"  Text length: {len(sample['text'])}")
        if 'entities' in sample:
            print(f"  Entities: {len(sample['entities'])}")
        
        # Teste transform
        transformed = scholar_xl_transform(sample)
        print(f"  After transform: {list(transformed.keys())}")

if __name__ == "__main__":
    inspect_scholar_xl_file("data/raw/train")
