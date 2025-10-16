import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
from src.eval_lstm import calculate_rouge_simple



def train_model_simple(model, train_loader, rouge_loader, tokenizer, epochs=10, learning_rate=0.001):
    """Упрощенная функция тренировки"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    rouge_history = []
    
    for epoch in range(epochs):
        # === ТРЕНИРОВКА ===
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            # Явно перемещаем данные на устройство
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # === ОЦЕНКА ROUGE ===
        print(f"\nEpoch {epoch+1} - Оценка ROUGE...")
        rouge_scores = calculate_rouge_simple(model, rouge_loader, tokenizer, device)  # Передаем device
        rouge_history.append(rouge_scores)
        
        print(f"\nEpoch {epoch+1}/{epochs} Результаты:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        for key, value in rouge_scores.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        # Сохраняем модель если ROUGE улучшился
        current_rouge1 = rouge_scores['rouge1']
        if current_rouge1 == max([rs.get('rouge1', 0) for rs in rouge_history]):
            torch.save(model.state_dict(), f'./models/best_model_epoch_{epoch+1}.pth')
            print("  ✅ Модель сохранена!")
    
    return model, train_losses, rouge_history

def plot_training_history(train_losses, rouge_history):
    """Визуализация результатов тренировки"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График потерь
    ax1.plot(train_losses, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='b')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # График ROUGE
    epochs = range(1, len(rouge_history) + 1)
    ax2.plot(epochs, [rs['rouge1'] for rs in rouge_history], 'r-', label='ROUGE-1', linewidth=2, marker='s')
    ax2.plot(epochs, [rs['rouge2'] for rs in rouge_history], 'g-', label='ROUGE-2', linewidth=2, marker='s')
    ax2.plot(epochs, [rs['rougeL'] for rs in rouge_history], 'b-', label='ROUGE-L', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROUGE Score')
    ax2.set_title('ROUGE Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    pass