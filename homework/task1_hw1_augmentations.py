import os
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset
from PIL import Image

# Указываем путь к папке train
train_dir = 'C:/Users/FuadI/OneDrive/Рабочий стол/homework/data/train'

# Определяем аугментации
transformations = {
    "HorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomCrop": transforms.RandomCrop(200),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    "RandomRotation": transforms.RandomRotation(45),
    "Grayscale": transforms.RandomGrayscale(p=1.0),
    "AllCombined": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(200),
        transforms.ColorJitter(),
        transforms.RandomRotation(45),
        transforms.RandomGrayscale(p=0.5),
        transforms.Resize((224, 224))
    ])
}

# Загружаем датасет без трансформаций
dataset = CustomImageDataset(root_dir=train_dir, transform=None)

# Выбираем по одному изображению из 5 разных классов
class_images = {}
print(f"Найдено {len(dataset.images)} изображений в train")
print(f"Классы: {dataset.classes}")
for img_path, label in zip(dataset.images, dataset.labels):
    class_name = dataset.classes[label]
    if class_name not in class_images:
        class_images[class_name] = img_path
    if len(class_images) == 6:
        break

# Функция визуализации
def show_augmentations(img_path, class_name):
    original_img = Image.open(img_path).convert('RGB').resize((224, 224))

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    axs[0].imshow(original_img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    for i, (name, transform) in enumerate(transformations.items()):
        if name == "AllCombined":
            img = transform(original_img)
        else:
            composed = transforms.Compose([transform, transforms.Resize((224, 224))])
            img = composed(original_img)
        axs[i+1].imshow(img)
        axs[i+1].set_title(name)
        axs[i+1].axis('off')

    plt.tight_layout()

    # Сохраняем результат
    save_path = os.path.join("result", f"{class_name}_augmentations.png")
    plt.savefig(save_path)
    print(f"💾 Сохранено: {save_path}")
    plt.close(fig)  # закрываем, чтобы не открывалось окно


# Визуализируем по одному изображению из каждого из 5 классов
for class_name, img_path in class_images.items():
    print(f"\n🔹 Класс: {class_name}")
    show_augmentations(img_path, class_name)

