import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
class RougeEvaluationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        
        for text in tqdm(texts, desc="Creating ROUGE evaluation samples"):
            if not text.strip():
                continue
                
            token_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,
                max_length=max_length, 
                truncation=True
            )
            
            if len(token_ids) < 20:  # Пропускаем слишком короткие тексты
                continue
                
            # Разделяем на 3/4 и 1/4
            split_point = int(len(token_ids) * 0.75)
            input_tokens = token_ids[:split_point]
            target_tokens = token_ids[split_point:]
            
            # Сохраняем и токены, и текст для ROUGE
            input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            self.samples.append((input_tokens, target_tokens, input_text, target_text))
           
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tokens, target_tokens, input_text, target_text = self.samples[idx]
        return torch.tensor(input_tokens), torch.tensor(target_tokens), input_text, target_text



class NextTokenPredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        
        for text in tqdm(texts, desc="Creating training samples"):
            if not text.strip():
                continue
                
            token_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,
                max_length=512, 
                truncation=True
            )
            
            if len(token_ids) <= seq_len:
                continue
                
            for i in range(len(token_ids) - seq_len):
                input_seq = token_ids[i:i + seq_len]
                target_token = token_ids[i + seq_len]
                self.samples.append((input_seq, target_token))
           
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class NextTokenDataset(Dataset):
    def __init__(self, texts: list, tokenizer, seq_len: int =7):
        self.samples = []
        self.tokenizer = tokenizer
        for text in tqdm(texts, desc="Preprocessing dataset", colour='green'):
            # Токенизируем текст
            token_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,  # Добавляем [CLS] и [SEP]
                max_length=512, 
                truncation=True
            )

            
            if len(token_ids) <= seq_len:
                continue
            
            for i in range(len(token_ids) - seq_len):
                # Входная последовательность: от i до i+seq_len
                input_seq = token_ids[i:i + seq_len]
                # Целевой токен: следующий после входной последовательности
                target_token = token_ids[i + seq_len]
                
                self.samples.append((input_seq, target_token))

    def __len__(self):
        return len(self.samples)

    def __vocab_size__(self):
        return self.tokenizer.vocab_size



    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
    


if __name__ == '__main__':

    pass



