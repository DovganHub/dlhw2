import evaluate
from transformers import pipeline

def calculate_rouge_transformer(generator, rouge_loader, num_examples=3):
    rouge_metric = evaluate.load("rouge")
    """Расчет ROUGE для модели distilgpt2 с использованием pipeline"""
    
    all_predictions = []
    all_references = []
    examples = []
    
    for i, (input_tokens, target_tokens, input_text, target_text) in enumerate(rouge_loader):
        if i >= 50:
            break
            
        # Используем max_new_tokens вместо max_length для избежания конфликтов
        target_length = len(target_text[0].split())  # примерная длина цели в словах
        max_new_tokens = min(target_length + 1, 100)  # ограничиваем максимальную длину
        
        result = generator(
            input_text[0],
            max_new_tokens=max_new_tokens,  # используем max_new_tokens
            do_sample=True,
            top_k=50,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id  # явно указываем pad_token_id
        )
        
        generated_text = result[0]["generated_text"]
        
        # Убираем оригинальный промпт из сгенерированного текста
        if generated_text.startswith(input_text[0]):
            generated_text = generated_text[len(input_text[0]):].strip()
        
        all_predictions.append(generated_text)
        all_references.append(target_text[0])
        
        if len(examples) < num_examples:
            examples.append({
                'input': input_text[0],
                'generated': generated_text,
                'target': target_text[0]
            })
    
    results = rouge_metric.compute(
        predictions=all_predictions, 
        references=all_references,
        use_stemmer=True
    )
    
    # Выводим примеры
    print("\n" + "="*80)
    print("ПРИМЕРЫ АВТОДОПОЛНЕНИЙ (distilgpt2):")
    print("="*80)
    for i, example in enumerate(examples, 1):
        print(f"\n--- Пример {i} ---")
        print(f"Вход (3/4 текста): {example['input'][:150]}...")
        print(f"Сгенерировано: {example['generated']}")
        print(f"Цель (1/4 текста): {example['target']}")
    
    return results