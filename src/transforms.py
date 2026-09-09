import albumentations as A


def get_train_transform(height=584, width=565):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(height=height, width=width),
        ],
        additional_targets={"fov": "mask"},
    )


def get_val_transform(height=584, width=565):
    return A.Compose(
        [A.Resize(height=height, width=width)],
        additional_targets={"fov": "mask"},
    )
