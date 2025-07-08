# Homework5
# Домашнее задание по обработке изображений и дообучению моделей

## 📁 Структура проекта
```bash
homework/
├── data/
│   ├── train/         
│   └── test/          
├── result/            
├── datasets.py        
├── task1_basic_augmentations.py
├── task2_custom_augmentations.py
├── task3_dataset_analysis.py
├── task4_augmentation_pipeline.py
├── task5_resize_experiment.py
├── finetune_resnet.py
├── README.md

---

## ✅ Задание 1: Базовые аугментации и визуализация (15 баллов)

**Файл:** `task1_hw1_augmentations.py`

- Применяются базовые аугментации из `torchvision.transforms`:
  - `RandomHorizontalFlip`
  - `RandomCrop`
  - `ColorJitter`
  - `RandomRotation`
  - `RandomGrayscale`
  - Комбинированная аугментация `AllCombined`
- Для 5 случайных классов выбирается по 1 изображению.
- Все аугментации отображаются в виде сетки 2x4.
- Полученные изображения сохраняются в `result/{classname}_augmentations.png`

---

## ✅ Задание 2: Кастомные аугментации (20 баллов)

**Файл:** `task2_compare_augmentations.py`

- Реализованы 3 кастомные аугментации:
  - Случайное размытие (`random_blur`)
  - Случайная перспектива (`random_perspective`)
  - Случайная яркость/контраст (`random_brightness_contrast`)
- Все аугментации применены к изображениям из `train`.
- Каждое изображение сохраняется в 4-х вариантах: оригинал + 3 аугментации.

---

## ✅ Задание 3: Анализ датасета (10 баллов)

**Файл:** `task3_dataset_analysis.py`

- Подсчитано количество изображений в каждом классе.
- Выведены минимальный, максимальный и средний размеры изображений.
- Построены:
  - Гистограмма распределения по классам.
  - Распределение размеров изображений.

---

## ✅ Задание 4: Pipeline аугментаций (20 баллов)

**Файл:** `task4_augmentation_pipeline.py`

- Реализован класс `AugmentationPipeline`:
  - `add_augmentation(name, func)`
  - `remove_augmentation(name)`
  - `apply(image)`
  - `get_augmentations()`
- Созданы 3 конфигурации:
  - **Light:** только размытие.
  - **Medium:** размытие + перспектива.
  - **Heavy:** размытие + перспектива + контраст.
- Применены ко всем изображениям (по 5 на класс).
- Сохраняются в `result/light/`, `result/medium/`, `result/heavy/`.

---

## ✅ Задание 5: Эксперимент с размерами (10 баллов)

**Файл:** `task5_experiment_resize`

- Измерено время и использование памяти при аугментации изображений размером:
  - `64x64`, `128x128`, `224x224`, `512x512`
- Обрабатываются по 100 изображений.
- Построены графики:
  - Время vs Размер
  - Память vs Размер

---

## ✅ Задание 6: Дообучение предобученных моделей (25 баллов)

**Файл:** `finetune_resnet.py`

- Использована предобученная `ResNet18` из `torchvision.models`.
- Последний слой заменён на количество классов в `train`.
- Обучение проводится на `train`, проверка — на `test`.
- Выводятся метрики по эпохам:
  - Train/Test loss
  - Train/Test accuracy
- Визуализируется график обучения.

---
