import os
import time
import tracemalloc
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random

input_dir = 'C:/Users/FuadI/OneDrive/Рабочий стол/homework/data/train'

# Простая аугментация (например, яркость + размытие)
def simple_augmentation(image):
    # Случайное размытие
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # Случайная яркость
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(random.uniform(0.7, 1.3))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Получить список файлов (до 100)
def get_image_files(path, max_files=100):
    files = []
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(class_path, filename))
                if len(files) >= max_files:
                    return files
    return files

sizes = [64, 128, 224, 512]

files = get_image_files(input_dir, max_files=100)
print(f"Используем {len(files)} изображений для эксперимента.")

times = []
memories = []

for size in sizes:
    print(f"\nРазмер: {size}x{size}")

    start_time = time.time()
    tracemalloc.start()

    for filepath in files:
        image_pil = Image.open(filepath).convert("RGB")
        image_pil = image_pil.resize((size, size))
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        _ = simple_augmentation(image)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = time.time() - start_time
    print(f"Время: {elapsed:.2f} сек, Пик памяти: {peak / 1024 / 1024:.2f} МБ")

    times.append(elapsed)
    memories.append(peak / 1024 / 1024)

# Визуализация
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(sizes, times, marker='o')
plt.title('Время выполнения аугментаций')
plt.xlabel('Размер изображения')
plt.ylabel('Время (сек)')

plt.subplot(1, 2, 2)
plt.plot(sizes, memories, marker='o', color='orange')
plt.title('Потребление памяти')
plt.xlabel('Размер изображения')
plt.ylabel('Память (МБ)')

plt.tight_layout()
plt.show()
