"""image generation for orthographic word stimuli."""

import math
import random
import shutil
from operator import mul
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

from config import DIR_DATA, DIR_FONTS, IMAGE_SIZE

_font_cache: dict = {}


def _load_font(font_name: str, size: int):
    """load a truetype font with caching to avoid repeated disk reads."""
    key = (font_name, size)
    if key not in _font_cache:
        _font_cache[key] = truetype(str(DIR_FONTS / font_name), size)
    return _font_cache[key]


def _letter_bbox(draw: Draw, char: str, font) -> tuple[int, int]:
    """compute width and height of a single character using textbbox."""
    bbox = draw.textbbox((0, 0), char, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


class CreateData:
    """create word images with configurable variation in font, rotation, and position."""

    def __init__(self, mode: str):
        self.mode = mode
        self.width: int = IMAGE_SIZE
        self.height: int = IMAGE_SIZE
        self.center: tuple = (self.width / 2, self.height / 2)
        self.ratio_train_valid_data: float = 4 / 1
        self.coefficient_data_valid: float = self.ratio_train_valid_data ** -1
        self.coefficient_data_train: float = 1 - self.coefficient_data_valid

    def set_attributes(self, c: dict):
        """set per-image attributes from a configuration dict."""
        if self.mode == "main":
            self.word = c["word"]
        else:
            self.target = c["target"]
            self.prime_type = c["prime"]
            self.content = c["content"]

        self.create_main_folders()
        self.name_font = c["font"]
        self.size_font = c["size"]
        self.position = c["position"]
        self.index = c["index"]
        self.name_save = c["name_save"]

    def set_configuration(self, configuration: dict):
        """set global variation parameters for image generation."""
        self.sd_rotation = configuration["standard_deviation_rotation"]
        self.variance_font = configuration["variance_font"]
        self.coefficient_space = configuration["coefficient_space"] + 1
        self.coefficient_translate = configuration["coefficient_translation"]

    def create_main_folders(self):
        """ensure output directory structure exists."""
        if self.mode == "main":
            (DIR_DATA / "data_all" / self.word).mkdir(parents=True, exist_ok=True)
            (DIR_DATA / "data_train").mkdir(parents=True, exist_ok=True)
            (DIR_DATA / "data_valid").mkdir(parents=True, exist_ok=True)
        else:
            (DIR_DATA / "prime_data" / self.target).mkdir(parents=True, exist_ok=True)

    def create_images(self):
        """render a single word image with configured variations."""
        self.word = self.content if self.mode == "prime" else self.word
        self.word = self.word.upper()
        canvas = new("RGB", (self.width, self.height), color=(0, 0, 0))
        draw = Draw(canvas)

        self.letters_size_font_shift = [
            random.randint(-self.variance_font, self.variance_font)
            for _ in range(len(self.word))
        ]

        letter_shapes = self._compute_letter_shapes(draw)
        self.width_letters_cumulative = self._cumulative_widths(letter_shapes)
        max_h = max(s[1] for s in letter_shapes)
        self.shape_half_word = (self.width_letters_cumulative[-1] / 2, max_h / 2)
        self.average_diagonal_length = sum(
            math.sqrt(s[0] ** 2 + s[1] ** 2) for s in letter_shapes
        ) / len(letter_shapes)
        self.radius_translate = self.coefficient_translate * self.average_diagonal_length
        self.coords_after_move = self._get_coords_after_move()

        for i in range(len(self.word)):
            letter_font = _load_font(self.name_font, self.size_font + self.letters_size_font_shift[i])
            bbox = draw.textbbox((0, 0), self.word[i], font=letter_font)
            shape_letter = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0, 0))
            Draw(canvas_letter).text((-bbox[0], -bbox[1]), self.word[i], fill=(255, 255, 255, 255), font=letter_font)

            angle = np.random.normal(loc=0.0, scale=self.sd_rotation)
            canvas_letter = self._rotate(canvas_letter, angle=angle)
            canvas.paste(im=canvas_letter, box=self._get_final_position_letter(i), mask=canvas_letter)

        save_path = self._get_save_path()
        canvas.save(save_path)

    def _compute_letter_shapes(self, draw: Draw) -> list[tuple[int, int]]:
        """compute bounding box dimensions for each letter in the word."""
        return [
            _letter_bbox(draw, self.word[i], _load_font(self.name_font, self.size_font + self.letters_size_font_shift[i]))
            for i in range(len(self.word))
        ]

    def _cumulative_widths(self, letter_shapes: list[tuple[int, int]]) -> np.ndarray:
        """compute cumulative x-offsets for letter placement."""
        widths = [s[0] * self.coefficient_space for s in letter_shapes]
        widths.insert(0, 0)
        return np.cumsum(widths)

    def _get_save_path(self) -> Path:
        """return the output file path based on mode."""
        if self.mode == "main":
            return DIR_DATA / "data_all" / self.word.lower() / self.name_save
        return DIR_DATA / "prime_data" / self.target / self.name_save

    def _rotate(self, image, angle: float):
        """rotate an image by a given angle in degrees."""
        return image.rotate(-angle, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC)

    def _get_translation_vector(self, radius: float) -> tuple:
        """compute a random 2d displacement within a circular radius."""
        r = radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        return (r * math.cos(theta), r * math.sin(theta))

    def _get_coords_after_move(self) -> tuple:
        """compute word origin after centering and positional offset."""
        word_position_vector = tuple(
            map(
                mul,
                self.position,
                (self.width * 0.5 - self.shape_half_word[0], self.height * 0.5 - self.shape_half_word[1]),
            )
        )
        coords_centered = tuple(np.subtract(self.center, self.shape_half_word))
        return tuple(map(sum, zip(word_position_vector, coords_centered)))

    def _get_final_position_letter(self, instance: int) -> tuple:
        """compute final pixel position for a letter including translation jitter."""
        position_letter = (
            self.coords_after_move[0] + self.width_letters_cumulative[instance],
            self.coords_after_move[1],
        )
        position_letter = tuple(map(sum, zip(self._get_translation_vector(self.radius_translate), position_letter)))
        return (int(position_letter[0]), int(position_letter[1]))

    def split_images_to_folders(self):
        """distribute generated images into train and valid folders."""
        data_all = DIR_DATA / "data_all"
        folders = [p for p in data_all.iterdir() if p.is_dir()]

        for folder in folders:
            images = [p.name for p in folder.iterdir() if p.is_file()]
            (DIR_DATA / "data_train" / folder.name).mkdir(parents=True, exist_ok=True)
            (DIR_DATA / "data_valid" / folder.name).mkdir(parents=True, exist_ok=True)

            train_images = set(random.sample(images, int(len(images) * self.coefficient_data_train)))
            valid_images = [img for img in images if img not in train_images]

            self._move_images(folder, DIR_DATA / "data_train" / folder.name, train_images)
            self._move_images(folder, DIR_DATA / "data_valid" / folder.name, valid_images)

        shutil.rmtree(data_all)

    def _move_images(self, src_folder: Path, dst_folder: Path, image_names):
        """move a collection of images from source to destination folder."""
        for img in image_names:
            shutil.move(src_folder / img, dst_folder / img)


def generate_single_image(args: tuple) -> None:
    """worker function for parallel image generation."""
    combo, mode, configuration = args
    creator = CreateData(mode=mode)
    creator.set_configuration(configuration)
    creator.set_attributes(combo)
    creator.create_images()
