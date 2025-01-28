import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Size:
    width: int
    height: int
    depth: int


@dataclass
class Bndbox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Part:
    name: str
    bndbox: Bndbox


@dataclass
class VOCObject:
    name: str
    pose: str
    truncated: bool
    difficult: int
    bndbox: Bndbox
    mask: np.array = None
    parts: List[Part] = None

    def __post_init__(self):
        if self.parts is None:
            self.parts = []


@dataclass
class VOCAnnotation:
    folder: str
    filename: str
    segmented: bool
    size: Size
    objects: List[VOCObject]
    image: np.array = None


def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    folder = root.find("folder").text
    filename = root.find("filename").text
    segmented = root.find("segmented").text == "1"

    size = root.find("size")
    size_obj = Size(
        width=int(size.find("width").text),
        height=int(size.find("height").text),
        depth=int(size.find("depth").text),
    )

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose = obj.find("pose").text
        truncated = obj.find("truncated").text == "1"
        difficult = int(obj.find("difficult").text)

        bndbox = obj.find("bndbox")
        bndbox_obj = Bndbox(
            xmin=int(bndbox.find("xmin").text),
            ymin=int(bndbox.find("ymin").text),
            xmax=int(bndbox.find("xmax").text),
            ymax=int(bndbox.find("ymax").text),
        )

        parts = []
        for part in obj.findall("part"):
            part_name = part.find("name").text
            part_bndbox = part.find("bndbox")
            part_bndbox_obj = Bndbox(
                xmin=int(part_bndbox.find("xmin").text),
                ymin=int(part_bndbox.find("ymin").text),
                xmax=int(part_bndbox.find("xmax").text),
                ymax=int(part_bndbox.find("ymax").text),
            )
            parts.append(Part(name=part_name, bndbox=part_bndbox_obj))

        obj_obj = VOCObject(
            name=name,
            pose=pose,
            truncated=truncated,
            difficult=difficult,
            bndbox=bndbox_obj,
            parts=parts,
        )
        objects.append(obj_obj)

    return VOCAnnotation(
        folder=folder,
        filename=filename,
        segmented=segmented,
        size=size_obj,
        objects=objects,
    )


def load_files_from_folder(folder_path):
    """Load all files from the folder."""
    folder = Path(folder_path)
    instance_mask_paths = list((folder / "SegmentationObject").rglob("*.png"))
    semantic_mask_paths = [
        mask_path.parent.parent / "SegmentationClass" / mask_path.name
        for mask_path in instance_mask_paths
    ]
    annotation_paths = [
        mask_path.parent.parent / "Annotations" / mask_path.with_suffix(".xml").name
        for mask_path in instance_mask_paths
    ]
    input_image_paths = [
        mask_path.parent.parent / "JPEGImages" / mask_path.with_suffix(".jpg").name
        for mask_path in instance_mask_paths
    ]

    return zip(
        instance_mask_paths, semantic_mask_paths, annotation_paths, input_image_paths
    )


def remove_colormap(filename):
    """Removes the color map from the annotation. https://stackoverflow.com/a/51675653"""
    return np.array(Image.open(filename)).astype(np.uint16)


def masks_to_unique_grayscale(semantic_mask_path, instance_mask_path):
    input_instance_mask = remove_colormap(instance_mask_path)
    input_semantic_mask = remove_colormap(semantic_mask_path)

    return (input_semantic_mask << 8) + input_instance_mask


def decode_value(value):
    labels = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
        255: "void",
    }

    semantic_id = int(value >> 8)
    instance_id = int(value & 0xFF)
    return labels[semantic_id], instance_id


def match_masks(voc_object, instance_mask_path, semantic_mask_path):
    # PASCAL has white borders, this approach is unnecessary and complex but safer. It can be used for other datasets.
    combined_mask = masks_to_unique_grayscale(semantic_mask_path, instance_mask_path)
    unique_values = np.unique(combined_mask)
    unique_values = unique_values

    for value in unique_values:
        mask = (combined_mask == value).astype(np.uint8) * 255
        class_name, instance_id = decode_value(value)

        if class_name in ["background", "void"]:
            continue

        for object in voc_object.objects:
            if class_name == object.name and object.mask is None:
                object.mask = mask
                break


def resize_and_pad(image, target_size=512, border_color=0):
    if len(image.shape) == 3:
        border_color = (
            border_color,
            border_color,
            border_color,
        )

    # Get the original dimensions
    h, w = image.shape[:2]

    # Determine the scaling factor
    scale = target_size / max(h, w)

    # Resize the image
    resized_image = cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )

    # Calculate padding
    new_h, new_w = resized_image.shape[:2]
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    # Add padding
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    # Ensure the image is exactly 512x512
    padded_image = cv2.resize(
        padded_image, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    return padded_image


def parse_voc(
    folder_path, remove_multiple_same_instance_images=True, allow_nonsquare_images=False
):
    for (
        instance_mask_path,
        semantic_mask_path,
        annotation_path,
        input_image_path,
    ) in load_files_from_folder(folder_path):
        input_annotation = parse_voc_xml(annotation_path)

        if remove_multiple_same_instance_images:
            # remove images with multiple same-type objects
            if len(input_annotation.objects) != len(
                set([obj.name for obj in input_annotation.objects])
            ):
                continue

        input_annotation.image = cv2.imread(str(input_image_path))
        match_masks(input_annotation, instance_mask_path, semantic_mask_path)

        if not allow_nonsquare_images:
            for obj in input_annotation.objects:
                if obj.mask is not None:
                    obj.mask = resize_and_pad(obj.mask)

            input_annotation.image = resize_and_pad(
                input_annotation.image, border_color=114
            )

            # TODO: BB and image size is wrong now but I don't use them

        yield input_annotation


if __name__ == "__main__":
    xml_file = "path_to_your_xml_file.xml"
    annotation = parse_voc_xml(xml_file)

    # Accessing the parsed data
    print(f"Folder: {annotation.folder}")
    print(f"Filename: {annotation.filename}")
    print(
        f"Size: {annotation.size.width}x{annotation.size.height}x{annotation.size.depth}"
    )
    for obj in annotation.objects:
        print(
            f"Object: {obj.name}, Pose: {obj.pose}, Bndbox: ({obj.bndbox.xmin}, {obj.bndbox.ymin}, {obj.bndbox.xmax}, {obj.bndbox.ymax})"
        )
        for part in obj.parts:
            print(
                f"Part: {part.name}, Bndbox: ({part.bndbox.xmin}, {part.bndbox.ymin}, {part.bndbox.xmax}, {part.bndbox.ymax})"
            )
