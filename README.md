/cheat-sheet-bot
│
├── app/                                # Логика приложения
│   ├── __init__.py                     # Инициализация пакета
│   ├── main.py                         # Основной запуск FastAPI
│   ├── routes.py                       # Маршруты API
│   ├── embeddings.py                   # Векторизация PDF и запись в Milvus
│   ├── generator.py                    # Генерация ответа через Qwen2-VL
│   ├── retriever.py                    # Поиск по Milvus и векторизация запросов
│   ├── converter.py                    # Конвертация PDF в изображения
│   └── agent.py                        # RAG-агент для обработки запросов
│
├── models/                             # Модели и промпты
│   ├── qwen2_vl.py                     # Настройка Qwen2-VL
│   ├── colpali.py                      # Настройка ColPali (мультимодальный ретривер)
│   └── prompts.py                      # Промпты для генерации и визуальных запросов
│
├── data/                               # Данные и векторное хранилище
│   ├── milvus/                         # Milvus коллекции и индексы
│   └── output/                         # Результаты работы
│
├── archive/                            # Датасет (PDF с cheat sheets)
│   ├── machine-learning/               # PDF по Machine Learning
│   │   ├── ml_cheat_sheet_1.pdf        
│   │   └── ml_cheat_sheet_2.pdf        
│   ├── sql/                            # PDF по SQL
│   └── python/                         # PDF по Python
│
├── configs/                            # Конфигурация
│   ├── config.yaml                     # Параметры моделей и Milvus
│   └── paths.yaml                      # Пути к индексам и датасету
│
├── scripts/                            # Вспомогательные скрипты
│   ├── ingest_to_milvus.py             # Загрузка PDF в Milvus
│   └── preprocess.py                   # Векторизация и предобработка PDF
│
├── requirements.txt                    # Зависимости проекта
└── README.md                           # Документация
