# qwen_cli.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig  # добавить в импорты
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.syntax import Syntax
from rich.prompt import Prompt

# from rich import print as rprint
import typer
import torch
import json
from pathlib import Path

console = Console()
app = typer.Typer()

DEFAULT_SYSTEM = (
    "Ты OCR-ассистент для распознавания ценников. "
    "Извлекай данные и возвращай ТОЛЬКО валидный JSON без пояснений:\n"
    '{"product_name":"...","price":"...","discount_percent":"...","product_code":"..."}\n'
    "Если поле не найдено — null."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

def load_model(model_id: str):
    console.rule("[bold cyan]Загрузка модели")
    console.print(f"[dim]{model_id}[/dim]")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    allocated = torch.cuda.memory_allocated() / 1e9
    console.print(f"[green]✓ Загружено[/green] VRAM: [yellow]{allocated:.2f} ГБ[/yellow]")
    return model, processor

def run_inference(model, processor, image_paths: list[str], system_prompt: str) -> str:
    content = []
    for p in image_paths:
        if not Path(p).exists():
            console.print(f"[red]Файл не найден: {p}[/red]")
            return ""
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": system_prompt})

    messages = [{"role": "user", "content": content}]
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
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def print_result(raw: str):
    # убрать <think>...</think> если есть (Qwen3)
    if "</think>" in raw:
        think, raw = raw.split("</think>", 1)
        with console.status(""):
            pass
        console.print(f"[dim]<think>{think}</think>[/dim]")
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        syntax = Syntax(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            "json",
            theme="monokai",
            line_numbers=False,
        )
        console.rule("[bold green]Результат")
        console.print(syntax)
    except json.JSONDecodeError:
        console.rule("[bold yellow]Сырой вывод (не JSON)")
        console.print(raw)


@app.command()
def main(
    model_id: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="HuggingFace model ID"
    ),
    system_prompt: str = typer.Option(
        DEFAULT_SYSTEM, "--system", "-s", help="Системный промпт"
    ),
):
    """
    Интерактивный CLI для инференса Qwen VL.
    Введи пути к изображениям через пробел, или 'exit' для выхода.
    Команды: :system — сменить промпт | :vram — показать память
    """
    model, processor = load_model(model_id)
    current_prompt = system_prompt

    console.rule("[bold]Готов к работе")
    console.print(
        "[dim]Введи пути к изображениям через пробел. "
        "Команды: [cyan]:system[/cyan] [cyan]:vram[/cyan] [cyan]exit[/cyan][/dim]\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Выход.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Выход.[/yellow]")
            break

        if user_input == ":system":
            console.print(f"[dim]Текущий промпт:[/dim]\n{current_prompt}\n")
            new = Prompt.ask("Новый системный промпт (Enter — оставить)")
            if new.strip():
                current_prompt = new.strip()
                console.print("[green]✓ Промпт обновлён[/green]")
            continue

        if user_input == ":vram":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            console.print(
                f"VRAM allocated: [yellow]{allocated:.2f} ГБ[/yellow]  "
                f"reserved: [yellow]{reserved:.2f} ГБ[/yellow]"
            )
            continue

        # остальное — пути к файлам
        paths = user_input.split()
        console.print(f"[dim]Обрабатываю {len(paths)} изображение(й)...[/dim]")

        with console.status("[bold green]Инференс...[/bold green]"):
            result = run_inference(model, processor, paths, current_prompt)

        if result:
            print_result(result)


if __name__ == "__main__":
    app()
