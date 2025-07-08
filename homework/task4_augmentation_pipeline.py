import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance

class AugmentationPipeline:
    def __init__(self):
        self.augs = {}

    def add_augmentation(self, name, aug):
        self.augs[name] = aug

    def remove_augmentation(self, name):
        if name in self.augs:
            del self.augs[name]

    def apply(self, image):
        results = {}
        for name, aug_func in self.augs.items():
            results[name] = aug_func(image.copy())
        return results

    def get_augmentations(self):
        return list(self.augs.keys())

# Аугментации
def blur(image):
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def perspective(image):
    h, w = image.shape[:2]
    margin = 60
    pts1 = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w - random.randint(0, margin), random.randint(0, margin)],
        [random.randint(0, margin), h - random.randint(0, margin)],
        [w - random.randint(0, margin), h - random.randint(0, margin)]
    ])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

def brightness_contrast(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer_b = ImageEnhance.Brightness(image_pil)
    enhancer_c = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer_b.enhance(random.uniform(0.7, 1.3))
    image_pil = enhancer_c.enhance(random.uniform(0.7, 1.3))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def rotate(image):
    h, w = image.shape[:2]
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))

def flip(image):
    return cv2.flip(image, 1)

# Пути к папкам — укажи свои
input_dir = 'C:/Users/FuadI/OneDrive/homework/data/train'
output_dir = 'C:/Users/FuadI/OneDrive/homework/result'

os.makedirs(output_dir, exist_ok=True)

# Конфигурации
pipelines = {}

pipeline_light = AugmentationPipeline()
pipeline_light.add_augmentation('blur', blur)
pipelines['light'] = pipeline_light

pipeline_medium = AugmentationPipeline()
pipeline_medium.add_augmentation('blur', blur)
pipeline_medium.add_augmentation('brightness_contrast', brightness_contrast)
pipelines['medium'] = pipeline_medium

pipeline_heavy = AugmentationPipeline()
pipeline_heavy.add_augmentation('blur', blur)
pipeline_heavy.add_augmentation('perspective', perspective)
pipeline_heavy.add_augmentation('brightness_contrast', brightness_contrast)
pipeline_heavy.add_augmentation('rotate', rotate)
pipeline_heavy.add_augmentation('flip', flip)
pipelines['heavy'] = pipeline_heavy

# Обработка, сохраняем по 5 файлов на класс на каждую конфигурацию
for config_name, pipeline in pipelines.items():
    print(f"Применяю конфигурацию: {config_name}")
    output_config_dir = os.path.join(output_dir, config_name)
    os.makedirs(output_config_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(output_config_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files = files[:5]
        for filename in files:
            input_path = os.path.join(class_path, filename)
            try:
                image_pil = Image.open(input_path).convert("RGB")
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                # Сохраняем оригинал
                orig_save_path = os.path.join(output_class_dir, f'orig_{filename}')
                cv2.imwrite(orig_save_path, image)
                print(f"Сохраняю оригинал: {orig_save_path}")

                # Применяем аугментации и сохраняем
                aug_images = pipeline.apply(image)
                for aug_name, aug_img in aug_images.items():
                    save_path = os.path.join(output_class_dir, f'{aug_name}_{filename}')
                    cv2.imwrite(save_path, aug_img)
                    print(f"Сохраняю аугментацию '{aug_name}': {save_path}")

                print(f"✅ Обработан файл: {os.path.join(class_name, filename)}")

            except Exception as e:
                print(f"❌ Ошибка при обработке {filename}: {e}")

print("\n🎉 Обработка завершена.")
