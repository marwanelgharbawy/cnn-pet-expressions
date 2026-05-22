from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



@dataclass(frozen=True)
class DataLoaderConfig:
	root_dir: str = os.path.join("dataset", "Master Folder")
	image_size: int = 224
	batch_size: int = 32
	num_workers: int = 0  # safer default on Windows/Jupyter
	pin_memory: bool = True
	persistent_workers: bool = False
	train_augmentation: bool = True
	normalize_imagenet: bool = True


def _build_transforms(
	image_size: int,
	train: bool,
	train_augmentation: bool,
	normalize_imagenet: bool,
) -> transforms.Compose:
	if normalize_imagenet:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)
	else:
		mean = (0.0, 0.0, 0.0)
		std = (1.0, 1.0, 1.0)

	if train:
		if train_augmentation:
			return transforms.Compose(
				[
					transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
					transforms.RandomHorizontalFlip(p=0.5),
					transforms.RandomRotation(degrees=15),
					transforms.ColorJitter(
						brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
					),
					transforms.ToTensor(),
					transforms.Normalize(mean=mean, std=std),
				]
			)
		return transforms.Compose(
			[
				transforms.Resize((image_size, image_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			]
		)

	return transforms.Compose(
		[
			transforms.Resize(image_size + 32),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def get_dataloaders(
	config: Optional[DataLoaderConfig] = None,
	*,
	seed: int = 42,
) -> Tuple[Dict[str, DataLoader], List[str]]:
	"""Create DataLoaders for dataset/Master Folder splits.

	Expected directory layout:
		dataset/Master Folder/
			train/<class_name>/*.jpg
			valid/<class_name>/*.jpg
			test/<class_name>/*.jpg

	Returns:
		(loaders, class_names)
	"""
	cfg = config or DataLoaderConfig()

	root = cfg.root_dir
	bs = cfg.batch_size
	size = cfg.image_size
	workers = cfg.num_workers

	split_dirs = {
		"train": os.path.join(root, "train"),
		"valid": os.path.join(root, "valid"),
		"test": os.path.join(root, "test"),
	}

	missing = [name for name, path in split_dirs.items() if not os.path.isdir(path)]
	if missing:
		raise FileNotFoundError(
			"Missing split folder(s) under root_dir. "
			f"root_dir='{root}', missing={missing}. "
			"Expected train/valid/test subfolders."
		)

	train_tfms = _build_transforms(
		size,
		train=True,
		train_augmentation=cfg.train_augmentation,
		normalize_imagenet=cfg.normalize_imagenet,
	)
	eval_tfms = _build_transforms(
		size,
		train=False,
		train_augmentation=False,
		normalize_imagenet=cfg.normalize_imagenet,
	)

	train_ds = datasets.ImageFolder(split_dirs["train"], transform=train_tfms)
	valid_ds = datasets.ImageFolder(split_dirs["valid"], transform=eval_tfms)
	test_ds = datasets.ImageFolder(split_dirs["test"], transform=eval_tfms)

	# Ensure consistent class ordering across splits.
	# ImageFolder sorts class folders alphabetically; if a split is missing a class, raise early.
	class_names = train_ds.classes
	for split_name, ds in [("valid", valid_ds), ("test", test_ds)]:
		if ds.classes != class_names:
			raise ValueError(
				"Class folders differ between splits. "
				f"train classes={class_names}, {split_name} classes={ds.classes}. "
				"Make sure every split has the same set of class subfolders."
			)

	generator = torch.Generator()
	generator.manual_seed(seed)

	common_loader_kwargs = dict(
		batch_size=bs,
		num_workers=workers,
		pin_memory=cfg.pin_memory and torch.cuda.is_available(),
		persistent_workers=cfg.persistent_workers and workers > 0,
		generator=generator,
	)

	loaders = {
		"train": DataLoader(train_ds, shuffle=True, **common_loader_kwargs),
		"valid": DataLoader(valid_ds, shuffle=False, **common_loader_kwargs),
		"test": DataLoader(test_ds, shuffle=False, **common_loader_kwargs),
	}

	return loaders, class_names


if __name__ == "__main__":
	loaders, class_names = get_dataloaders()
	print("Classes:", class_names)
	xb, yb = next(iter(loaders["train"]))
	print("Batch x:", tuple(xb.shape), "Batch y:", tuple(yb.shape))
