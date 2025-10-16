Проект "Автодополнение текстов"

text-autocomplete/
├── data/                            # Датасеты
│   ├── raw_dataset.csv              # "сырой" скачанный датасет

│
├── src/                             # Весь код проекта
│   ├── data_utils.py                # Обработка датасета
│   ├── lstm_model.py                # код lstm модели
|   ├── eval_lstm.py                 # замер метрик lstm модели
|   ├── lstm_train.py                # код обучения модели
|   ├── eval_transformer_pipeline.py # код с запуском и замером качества трансформера
│
├── configs/                         # yaml-конфиги с настройками проекта
│
├── models/                          # веса обученных моделей
|
├── solution.ipynb                   # ноутбук с решением
├── requirements.txt                 # зависимости проекта
└── eda.ipynb                        # eda (распределения длин последовательностей)
