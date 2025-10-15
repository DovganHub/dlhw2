import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import evaluate


def calculate_rouge_simple(model, rouge_loader, tokenizer, device, num_examples=3):
    rouge_metric = evaluate.load("rouge")
    """Упрощенный расчет ROUGE с использованием библиотеки evaluate"""
    model.eval()
    
    all_predictions = []
    all_references = []
    examples = []
    
    with torch.no_grad():
        for i, (input_tokens, target_tokens, input_text, target_text) in enumerate(rouge_loader):
            if i >= 50:  # Оцениваем только на 50 примерах для скорости
                break
                
            # Перемещаем тензоры на правильное устройство
            input_tokens = input_tokens.to(device)
            
            # Генерируем продолжение с указанием устройства
            generated_text = model.generate(
                tokenizer=tokenizer,
                prompt=input_text[0],
                max_length=min(len(target_tokens[0]) + 10, 100),  # Ограничиваем максимальную длину
                temperature=0.8,
                top_k=50,
                device=device  # Добавляем параметр устройства
            )
            
            all_predictions.append(generated_text)
            all_references.append(target_text[0])
            
            # Сохраняем примеры для вывода
            if len(examples) < num_examples:
                examples.append({
                    'input': input_text[0],
                    'generated': generated_text,
                    'target': target_text[0]
                })
    
    # Вычисляем ROUGE с помощью библиотеки evaluate
    results = rouge_metric.compute(
        predictions=all_predictions, 
        references=all_references,
        use_stemmer=True
    )
    
    # Выводим примеры
    print("\n" + "="*80)
    print("ПРИМЕРЫ АВТОДОПОЛНЕНИЙ:")
    print("="*80)
    for i, example in enumerate(examples, 1):
        print(f"\n--- Пример {i} ---")
        print(f"Вход (3/4 текста): {example['input'][:150]}...")
        print(f"Сгенерировано: {example['generated']}")
        print(f"Цель (1/4 текста): {example['target']}")
    
    return results

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



if __name__ == '__main__':
    pass