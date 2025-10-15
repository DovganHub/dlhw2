import torch
from torch import nn
class LSTMForNextTokenPrediction(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        if lstm_out.size(1) > 0:
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, 0, :]
        output = self.fc(last_output)
        return output, hidden
    
    def init_hidden(self, batch_size=1, device='cpu'):
        """Инициализация hidden state на правильном устройстве"""
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
    
    def generate(self, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, device='cpu'):
        """Генерация текста на основе промпта"""
        self.eval()
        
        # Токенизируем промпт и перемещаем на правильное устройство
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
        generated = input_ids.clone()
        
        # Инициализируем hidden state на правильном устройстве
        hidden = self.init_hidden(batch_size=1, device=device)
        
        with torch.no_grad():
            # Пропускаем промпт через модель для получения начального состояния
            if input_ids.size(1) > 0:
                _, hidden = self(input_ids, hidden)
            
            for _ in range(max_length):
                if generated.size(1) > 0:
                    last_token = generated[:, -1:]
                else:
                    # Если промпт пустой, начинаем со случайного токена на правильном устройстве
                    last_token = torch.randint(0, self.vocab_size, (1, 1)).to(device)
                
                outputs, hidden = self(last_token, hidden)
                next_token_logits = outputs / temperature
                
                # Top-k фильтрация
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Останавливаемся на [SEP] токене или если текст слишком длинный
                if next_token.item() == tokenizer.sep_token_id or generated.size(1) > max_length + 20:
                    break
        
        # Перемещаем обратно на CPU для декодирования
        generated_text = tokenizer.decode(generated.cpu()[0].tolist(), skip_special_tokens=True)
        return generated_text
    

if __name__ == '__main__':

    pass