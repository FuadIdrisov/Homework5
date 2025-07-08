import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance

def random_blur(image):
    # Применяем всегда с более сильным размытием
    ksize = 5
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def random_perspective(image):
    h, w = image.shape[:2]
    margin = 100  # больше сдвиг
    pts1 = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],
        [w - random.randint(0, margin), random.randint(0, margin)],
        [random.randint(0, margin), h - random.randint(0, margin)],
        [w - random.randint(0, margin), h - random.randint(0, margin)]
    ])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

def random_brightness_contrast(image):
    # Применяем всегда с более выраженным эффектом
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer_b = ImageEnhance.Brightness(image_pil)
    enhancer_c = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer_b.enhance(1.7)  # сильнее яркость
    image_pil = enhancer_c.enhance(1.7)  # сильнее контраст
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Функции resize_image_by_height и add_label_below — без изменений, как в предыдущем коде

def resize_image_by_height(image, target_height):
    h, w = image.shape[:2]
    new_w = int(w * target_height / h)
    resized_img = cv2.resize(image, (new_w, target_height))
    return resized_img

def add_label_below(image, label, font_scale=0.7, thickness=2):
    text_height = 30
    h, w = image.shape[:2]
    labeled_img = np.ones((h + text_height, w, 3), dtype=np.uint8) * 255
    labeled_img[0:h, 0:w, :] = image
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + (text_height + text_size[1]) // 2
    cv2.putText(labeled_img, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, lineType=cv2.LINE_AA)
    return labeled_img

# Основной код без изменений, только убрал проверку вероятности

input_dir = 'C:/Users/FuadI/OneDrive/Рабочий стол/homework/data/train'
target_height = 300

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    filenames = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not filenames:
        print(f"❌ В папке {class_name} нет изображений")
        continue

    filename = filenames[0]
    input_path = os.path.join(class_path, filename)
    print(f"🔍 Обрабатываю файл: {input_path}")

    try:
        image_pil = Image.open(input_path).convert("RGB")
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        if image is None or image.size == 0:
            print(f"❌ Пустое изображение, пропуск: {filename}")
            continue

        aug1 = random_blur(image.copy())
        aug2 = random_perspective(image.copy())
        aug3 = random_brightness_contrast(image.copy())

        imgs = [image, aug1, aug2, aug3]
        labels = ['Original', 'Blur', 'Perspective', 'Brightness/Contrast']

        labeled_imgs = []
        for img, label in zip(imgs, labels):
            resized = resize_image_by_height(img, target_height)
            labeled = add_label_below(resized, label)
            labeled_imgs.append(labeled)

        combined = cv2.hconcat(labeled_imgs)

        window_title = f'Hero: {class_name} - Original + Augmentations'
        cv2.imshow(window_title, combined)
        print(f"✅ Аугментации показаны для класса: {class_name}")
        print("Нажмите любую клавишу для продолжения...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Ошибка при обработке {filename}: {e}")

print("\n🎉 Готово! Все герои обработаны и показаны.")
