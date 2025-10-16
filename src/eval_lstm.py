import evaluate
import torch

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