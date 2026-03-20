"""tests for image generation: transparent-bg optimisation and multiprocessing."""

import shutil
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw

from config import DIR_DATA, DIR_FONTS, IMAGE_SIZE
from utils.data_generate.main import CreateData, generate_single_image, _load_font


TEST_DIR = DIR_DATA / "_test_generation"

PRIME_CONFIG = {
    "standard_deviation_rotation": 0,
    "coefficient_translation": 0,
    "coefficient_space": 0.5,
    "variance_font": 0,
}

TRAIN_CONFIG = {
    "standard_deviation_rotation": 8.0,
    "coefficient_translation": 0.04,
    "coefficient_space": 0.5,
    "variance_font": 3,
}

SAMPLE_PRIME_COMBO = {
    "target": "design",
    "prime": "ID",
    "content": "design",
    "font": "arial.ttf",
    "size": 22,
    "position": (0, 0),
    "index": 0,
    "name_save": "ID.png",
}


def _cleanup():
    """remove test output directory."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def _render_old_method(word: str, font_name: str, size: int) -> Image.Image:
    """render using the old black-bg + pixel-loop transparency method."""
    canvas = new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    draw = Draw(canvas)
    font = _load_font(font_name, size)

    bbox = draw.textbbox((0, 0), word, font=font)
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    canvas_letter = new("RGBA", shape, color=(0, 0, 0))
    Draw(canvas_letter).text((-bbox[0], -bbox[1]), word, fill=(255, 255, 255, 255), font=font)
    canvas_letter.putdata([(0, 0, 0, 0) if px[:3] == (0, 0, 0) else px for px in canvas_letter.getdata()])
    canvas.paste(im=canvas_letter, box=(IMAGE_SIZE // 2 - shape[0] // 2, IMAGE_SIZE // 2 - shape[1] // 2), mask=canvas_letter)
    return canvas


def _render_new_method(word: str, font_name: str, size: int) -> Image.Image:
    """render using the new transparent-bg method (no pixel loop)."""
    canvas = new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    draw = Draw(canvas)
    font = _load_font(font_name, size)

    bbox = draw.textbbox((0, 0), word, font=font)
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    canvas_letter = new("RGBA", shape, color=(0, 0, 0, 0))
    Draw(canvas_letter).text((-bbox[0], -bbox[1]), word, fill=(255, 255, 255, 255), font=font)
    canvas.paste(im=canvas_letter, box=(IMAGE_SIZE // 2 - shape[0] // 2, IMAGE_SIZE // 2 - shape[1] // 2), mask=canvas_letter)
    return canvas


def test_transparent_bg_identical_no_rotation():
    """unrotated rendering: transparent-bg must be pixel-identical to old pixel-loop."""
    words = ["DESIGN", "HELLO", "A", "XYZ", "BREAKING"]
    fonts = [f.name for f in DIR_FONTS.iterdir() if f.suffix == ".ttf"][:3]

    for word in words:
        for font in fonts:
            old = np.array(_render_old_method(word, font, 22))
            new_img = np.array(_render_new_method(word, font, 22))
            assert old.shape == new_img.shape, f"shape mismatch: {word}/{font}"
            assert np.array_equal(old, new_img), f"pixel mismatch: {word}/{font}"

    print(f"pass: {len(words) * len(fonts)} image pairs pixel-identical")


def test_rotation_produces_valid_output():
    """rotated rendering with PIL rotate should produce valid letter images."""
    font_name = "arial.ttf"
    font = _load_font(font_name, 22)
    char = "D"

    bbox = Draw(new("RGBA", (1, 1))).textbbox((0, 0), char, font=font)
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    canvas_letter = new("RGBA", shape, color=(0, 0, 0, 0))
    Draw(canvas_letter).text((-bbox[0], -bbox[1]), char, fill=(255, 255, 255, 255), font=font)
    rotated = canvas_letter.rotate(-15.0, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC)

    final = new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    final.paste(rotated, box=(50, 50), mask=rotated)

    arr = np.array(final)
    assert arr.max() > 200, "rotated letter has no bright pixels"
    assert (arr > 0).sum() > 20, "rotated letter has too few visible pixels"
    white_pixels_original = (np.array(canvas_letter)[:, :, 3] > 0).sum()
    white_pixels_rotated = (np.array(rotated)[:, :, 3] > 0).sum()
    assert white_pixels_rotated >= white_pixels_original * 0.8, "rotation lost too many pixels"
    print(f"pass: PIL rotation produces valid output ({white_pixels_rotated} visible pixels, max={arr.max()})")


def test_prime_generation_serial_vs_parallel():
    """prime images from serial and parallel generation must be pixel-identical."""
    _cleanup()
    test_words = ["design", "bridge", "castle"]

    combos = [
        {"target": w, "prime": "ID", "content": w, "font": "arial.ttf",
         "size": 22, "position": (0, 0), "index": 0, "name_save": "ID.png"}
        for w in test_words
    ]

    creator = CreateData(mode="prime")
    creator.set_configuration(PRIME_CONFIG)
    for combo in combos:
        creator.set_attributes(combo)
        creator.create_images()

    serial_images = {}
    for combo in combos:
        path = DIR_DATA / "prime_data" / combo["target"] / "ID.png"
        serial_images[combo["target"]] = np.array(Image.open(path))
        path.unlink()

    args = [(c, "prime", PRIME_CONFIG) for c in combos]
    with Pool(cpu_count()) as pool:
        pool.map(generate_single_image, args)

    mismatches = 0
    for combo in combos:
        path = DIR_DATA / "prime_data" / combo["target"] / "ID.png"
        parallel_img = np.array(Image.open(path))
        serial_img = serial_images[combo["target"]]
        if not np.array_equal(serial_img, parallel_img):
            mismatches += 1

    for w in test_words:
        shutil.rmtree(DIR_DATA / "prime_data" / w, ignore_errors=True)
    _cleanup()
    assert mismatches == 0, f"{mismatches} images differ between serial and parallel"
    print(f"pass: {len(combos)} prime images identical between serial and parallel")


def test_image_properties():
    """generated images must have correct dimensions and contain white pixels."""
    creator = CreateData(mode="prime")
    creator.set_configuration(PRIME_CONFIG)
    creator.set_attributes(SAMPLE_PRIME_COMBO)
    creator.create_images()

    path = DIR_DATA / "prime_data" / "design" / "ID.png"
    img = Image.open(path)
    arr = np.array(img)

    assert img.size == (IMAGE_SIZE, IMAGE_SIZE), f"expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {img.size}"
    assert arr.max() > 0, "image is entirely black"
    has_white = (arr > 200).any()
    assert has_white, "no white pixels found (text not rendered)"
    print(f"pass: image is {img.size}, contains text (max pixel = {arr.max()})")

    shutil.rmtree(DIR_DATA / "prime_data" / "design", ignore_errors=True)


def test_multiprocessing_speedup():
    """benchmark: parallel generation should be faster than serial."""
    fonts = [f.name for f in DIR_FONTS.iterdir() if f.suffix == ".ttf"][:3]
    words = ["design", "bridge", "castle", "museum", "throne"]
    combos = [
        {"target": w, "prime": "ID", "content": w, "font": font,
         "size": 22, "position": (0, 0), "index": 0, "name_save": f"{font}.png"}
        for w in words for font in fonts
    ]

    _cleanup()

    t0 = time.perf_counter()
    creator = CreateData(mode="prime")
    creator.set_configuration(PRIME_CONFIG)
    for c in combos:
        creator.set_attributes(c)
        creator.create_images()
    serial_time = time.perf_counter() - t0

    for w in words:
        shutil.rmtree(DIR_DATA / "prime_data" / w, ignore_errors=True)

    t0 = time.perf_counter()
    args = [(c, "prime", PRIME_CONFIG) for c in combos]
    with Pool(cpu_count()) as pool:
        pool.map(generate_single_image, args)
    parallel_time = time.perf_counter() - t0

    for w in words:
        shutil.rmtree(DIR_DATA / "prime_data" / w, ignore_errors=True)

    speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
    print(f"benchmark: serial={serial_time:.3f}s, parallel={parallel_time:.3f}s, speedup={speedup:.1f}x ({cpu_count()} cores)")

    _cleanup()


def test_pil_rotate_matches_reference():
    """prime images (no rotation) must be pixel-identical to scipy-generated reference."""
    ref_dir = DIR_DATA / "_reference"
    if not ref_dir.exists():
        print("skip: no reference images found (generate with scipy first)")
        return

    words = [p.stem for p in ref_dir.glob("*.png")]
    creator = CreateData(mode="prime")
    creator.set_configuration(PRIME_CONFIG)

    mismatches = 0
    for w in words:
        combo = {"target": w, "prime": "ID", "content": w, "font": "arial.ttf",
                 "size": 22, "position": (0, 0), "index": 0, "name_save": "ID.png"}
        creator.set_attributes(combo)
        creator.create_images()

        ref = np.array(Image.open(ref_dir / f"{w}.png"))
        new_img = np.array(Image.open(DIR_DATA / "prime_data" / w / "ID.png"))
        if not np.array_equal(ref, new_img):
            mismatches += 1

    for w in words:
        shutil.rmtree(DIR_DATA / "prime_data" / w, ignore_errors=True)

    assert mismatches == 0, f"{mismatches}/{len(words)} images differ from scipy reference"
    print(f"pass: {len(words)} images pixel-identical to scipy reference (no rotation)")


if __name__ == "__main__":
    test_transparent_bg_identical_no_rotation()
    test_rotation_produces_valid_output()
    test_image_properties()
    test_pil_rotate_matches_reference()
    test_prime_generation_serial_vs_parallel()
    test_multiprocessing_speedup()
    print("\nall tests passed")
