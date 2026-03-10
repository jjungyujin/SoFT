import json
import os
from pathlib import Path
import sys
from typing import Optional, Union, Literal

import PIL
import PIL.Image

import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).parent.parent))

from llm.utils import ROOT_PATH

MULTI_TARGET_DIR = ROOT_PATH / "mt_pipeline" / "llm" / "multi_target_selection"
SINGLE_TARGET_DIR = ROOT_PATH / "mt_pipeline" / "llm" / "single_target_selection"

DRESSTYPE2MAXNUMGTS = {
    "dress": 18,
    "shirt": 20,
    "toptee": 16,
}
CIRR_MAX_NUM_GTS = 16


class MultiTargetFashionIQDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        target_type: Literal["multi", "single"],
        dress_type: Literal["dress", "toptee", "shirt"],
        mode: Literal["relative", "classic"],
        preprocess: callable,
        blip_transform: Optional[callable] = None,
        split="val",
    ):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.target_type = target_type
        self.dress_type = dress_type
        self.split = split
        self.blip_transform = blip_transform

        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")
        if dress_type not in ["dress", "shirt", "toptee"]:
            raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")
        if target_type not in ["multi", "single"]:
            raise ValueError("target_type should be in ['multi', 'single']")
        if split not in ["val"]:
            raise ValueError("split should be in ['val']")

        self.preprocess = preprocess
        with open(
            dataset_path / "image_splits" / f"split.{dress_type}.{split}.json"
        ) as f:
            self.image_names = json.load(f)

        if target_type == "multi":
            with open(
                Path(MULTI_TARGET_DIR) / f"{dress_type}_multi_target_selection.json"
            ) as f:
                self.triplets = json.load(f)
            self.max_num_gts = DRESSTYPE2MAXNUMGTS[dress_type]

        else:
            with open(
                Path(SINGLE_TARGET_DIR)
                / f"{dress_type}_selection_single_target_seed42.json"
            ) as f:
                self.triplets = json.load(f)
            self.max_num_gts = 1

        print(
            f"MultiTargetFashionIQ {split} - {dress_type} {target_type} dataset in {mode} mode initialized"
        )

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == "relative":
                triplet_dict = self.triplets[index]

                if self.target_type == "multi":
                    relative_captions = triplet_dict["relative_captions"]
                    target_names = list(triplet_dict["confidence_scores"].keys())

                elif self.target_type == "single":
                    relative_captions = triplet_dict["refined_caption"]
                    target_names = [triplet_dict["target_image_name"]]

                else:
                    raise ValueError("target_type should be in ['multi', 'single']")

                reference_name = triplet_dict["reference_image_name"]
                reference_image_path = (
                    self.dataset_path / "images" / f"{reference_name}.png"
                )
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                target_images = []
                actual_num_targets = len(target_names)

                if self.blip_transform is not None:
                    blip_ref_img = self.blip_transform(
                        PIL.Image.open(reference_image_path).convert("RGB")
                    )
                    blip_target_imgs = []

                for i in range(self.max_num_gts):
                    if i < actual_num_targets:
                        target_name = target_names[i]
                        target_image_path = (
                            self.dataset_path / "images" / f"{target_name}.png"
                        )
                        target_image = self.preprocess(
                            PIL.Image.open(target_image_path)
                        )
                        if self.blip_transform is not None:
                            blip_target_img = self.blip_transform(
                                PIL.Image.open(target_image_path).convert("RGB")
                            )
                    else:
                        empty_image = PIL.Image.new("RGB", (224, 224), (255, 255, 255))
                        target_image = self.preprocess(empty_image)
                        if self.blip_transform is not None:
                            blip_target_img = self.blip_transform(empty_image)

                    target_images.append(target_image)
                    if self.blip_transform is not None:
                        blip_target_imgs.append(blip_target_img)

                target_names += [""] * (self.max_num_gts - actual_num_targets)

                item_dict = {
                    "reference_name": reference_name,
                    "reference_image": reference_image,
                    "target_name": tuple(target_names),
                    "target_image": torch.stack(target_images),
                    "relative_captions": relative_captions,
                }
                if self.blip_transform is not None:
                    item_dict.update(
                        {
                            "blip_ref_img": blip_ref_img,
                            "blip_target_img": torch.stack(blip_target_imgs),
                        }
                    )

                return item_dict

            elif self.mode == "classic":
                image_name = self.image_names[index]
                image_path = self.dataset_path / "images" / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                if self.blip_transform is not None:
                    blip_img = self.blip_transform(
                        PIL.Image.open(image_path).convert("RGB")
                    )
                item_dict = {
                    "image": image,
                    "image_name": image_name,
                }
                if self.blip_transform is not None:
                    item_dict.update({"blip_img": blip_img})
                return item_dict
            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class MultiTargetCIRRDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        target_type: Literal["multi", "single"],
        mode: Literal["relative", "classic"],
        preprocess: callable,
        blip_transform: callable = None,
        split="val",
    ):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.target_type = target_type
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.blip_transform = blip_transform

        if split not in ["val"]:
            raise ValueError("split should be in ['val']")
        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")
        if target_type not in ["multi", "single"]:
            raise ValueError("target_type should be in ['multi', 'single']")

        with open(
            dataset_path / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        ) as f:
            self.name_to_relpath = json.load(f)

        if target_type == "multi":
            with open(
                Path(MULTI_TARGET_DIR) / f"cirr_multi_target_selection.json"
            ) as f:
                self.triplets = json.load(f)
            self.max_num_gts = CIRR_MAX_NUM_GTS

        else:
            with open(
                Path(SINGLE_TARGET_DIR) / f"cirr_selection_single_target_seed42.json"
            ) as f:
                self.triplets = json.load(f)
            self.max_num_gts = 1

        print(
            f"MultiTargetCIRR {split} {target_type} dataset in {mode} mode initialized"
        )

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == "relative":
                triplet_dict = self.triplets[index]

                if self.target_type == "multi":
                    relative_caption = triplet_dict["relative_captions"]
                    target_names = list(triplet_dict["confidence_scores"].keys())

                elif self.target_type == "single":
                    relative_caption = triplet_dict["refined_caption"]
                    target_names = [triplet_dict["target_image_name"]]

                else:
                    raise ValueError("target_type should be in ['multi', 'single']")

                reference_name = triplet_dict["reference_image_name"]
                reference_image_path = (
                    self.dataset_path / self.name_to_relpath[reference_name]
                )
                reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                if self.blip_transform is not None:
                    blip_ref_img = self.blip_transform(
                        PIL.Image.open(reference_image_path).convert("RGB")
                    )

                actual_num_targets = len(target_names)
                target_images = []
                for i in range(self.max_num_gts):
                    if i < actual_num_targets:
                        target_name = target_names[i]
                        target_image_path = (
                            self.dataset_path / self.name_to_relpath[target_name]
                        )
                        target_image = self.preprocess(
                            PIL.Image.open(target_image_path)
                        )
                    else:
                        empty_image = PIL.Image.new("RGB", (224, 224), (255, 255, 255))
                        target_image = self.preprocess(empty_image)
                    target_images.append(target_image)

                target_names += [""] * (self.max_num_gts - actual_num_targets)

                item_dict = {
                    "reference_name": reference_name,
                    "reference_image": reference_image,
                    "target_name": tuple(target_names),
                    "target_image": torch.stack(target_images),
                    "relative_caption": relative_caption,
                }
                if self.blip_transform is not None:
                    item_dict.update({"blip_ref_img": blip_ref_img})
                return item_dict

            elif self.mode == "classic":
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.dataset_path / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return {"image": image, "image_name": image_name}
            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
