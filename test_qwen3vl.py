from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import argparse
import torch
import json
import sys

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # старт с 2.5-VL-7B — стабильнее
# MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # раскомментировать после теста
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

model = None
processor = None


def load_model() -> None:
    global model, processor

    if model is not None and processor is not None:
        return

    print("Загрузка модели...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Файл не похож на изображение: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Путь не найден: {input_path}")

    paths = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract(image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        "Это ценник из магазина. "
                        "Извлеки данные и верни ТОЛЬКО JSON без пояснений:\n"
                        '{"product_name":"...","price":"...",'
                        '"discount_percent":"...","product_code":"..."}\n'
                        "Если поле не найдено — null."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = out[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запустить Qwen-VL для одного изображения или папки с изображениями."
    )
    parser.add_argument("path", help="Путь к картинке или папке с картинками")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Искать картинки во вложенных папках",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        image_paths = collect_images(Path(args.path), args.recursive)
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    if not image_paths:
        print(
            f"В папке нет картинок с расширениями: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
        return 1

    print(f"Найдено изображений: {len(image_paths)}")
    load_model()

    for index, path in enumerate(image_paths, start=1):
        print(f"\n[{index}/{len(image_paths)}] {path}")
        result = extract(str(path))
        print("Результат:", result)

        try:
            parsed = json.loads(result)
            print("Parsed OK:", json.dumps(parsed, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print("Предупреждение: ответ не является валидным JSON")

    return 0


if __name__ == "__main__":
    sys.exit(main())
