"""A tiny random-weight Qwen2 causal LM and a toy byte-level BPE tokenizer, built offline for tests.

The tokenizer uses the same normaliser and pre-tokeniser as ``Qwen2Tokenizer`` (NFC, the Qwen split
regex, byte-level), so ``AutoTokenizer.from_pretrained`` (which rebuilds a ``Qwen2Tokenizer`` from the
vocabulary and merges because ``config.json`` says ``model_type: qwen2``) tokenises exactly like the
tokenizer that was trained here. Special tokens and the chat template follow Qwen2.5-Instruct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")  # ids 0, 1, 2
PAD_ID, IM_START_ID, IM_END_ID = 0, 1, 2

QWEN_SPLIT_REGEX = (  # transformers.models.qwen2.tokenization_qwen2.PRETOKENIZE_REGEX
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if loop.first and message['role'] != 'system' %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

CORPUS = (
    "You are a helpful assistant.",
    "Classify the sentiment of the review into exactly one label.",
    "Classify the customer's banking message into exactly one intent label.",
    "Extract the invoice fields as JSON matching the schema.",
    "I loved this product, it works great!",
    "Terrible quality, broke after one day.",
    "It is okay, nothing special.",
    "My card has not arrived yet. When will my new card arrive?",
    "positive",
    "negative",
    "neutral",
    "mixed feelings",
    "card",
    "card arrival",
    "cash withdrawal",
    "system",
    "user",
    "assistant",
    '{"label": "positive"}',
    '{"vendor_name": "Example Tools Ltd", "invoice_number": "INV-2024-0042", "total_amount": 1234.5}',
    '{"invoice_date": "2024-03-01", "due_date": null, "currency": "EUR", "tax_amount": 12.34}',
    "Invoice INV-2024-0042 from Example Tools Ltd, total EUR 1.234,56, due in 30 days.",
    "Please find attached the invoice. Contact billing@example.com or call 555-0123.",
    "0123456789 .,:;!?-_/()[]{}\"'",
)


def build_tokenizer(vocab_size: int = 512) -> Any:
    """Train the toy tokenizer (deterministic for a fixed corpus) and wrap it for transformers."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_SPLIT_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(list(CORPUS) * 8, trainer=trainer)
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIAL_TOKENS])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
        model_max_length=2048,
        model_input_names=["input_ids", "attention_mask"],
    )


def build_tiny_model(
    directory: Path | str,
    *,
    seed: int = 0,
    hidden_size: int = 64,
    num_layers: int = 2,
    initializer_range: float = 0.2,
) -> Path:
    """Write a tiny random-weight ``Qwen2ForCausalLM`` plus the toy tokenizer into ``directory``.

    No downloads; takes well under a second. The weights depend only on ``seed`` (the global torch
    RNG state is restored afterwards). ``initializer_range`` is larger than Qwen's 0.02 so that
    LoRA on such a small model can move the logits within a few steps.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=IM_END_ID,
        pad_token_id=PAD_ID,
        initializer_range=initializer_range,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(config)
    model.generation_config.eos_token_id = IM_END_ID
    model.generation_config.pad_token_id = PAD_ID
    model.generation_config.do_sample = False
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


TOY_LABELS = ("card", "card arrival", "cash withdrawal")  # "card" is a token prefix of "card arrival"
TOY_SYSTEM_PROMPT = "Classify the customer's banking message into exactly one intent label."
_TOY_MESSAGES = {
    "card": ("My card is broken", "The card was declined", "Please block my card", "The card chip is damaged"),
    "card arrival": ("My new card has not arrived", "When will my card arrive", "Card still not delivered"),
    "cash withdrawal": ("I want to withdraw cash", "The ATM kept my cash", "Cash withdrawal limit today"),
}


def toy_example(i: int) -> tuple[str, str]:
    """The ``i``-th toy banking message and its label (deterministic, cycles through the labels)."""
    label = TOY_LABELS[i % len(TOY_LABELS)]
    texts = _TOY_MESSAGES[label]
    return f"{texts[(i // len(TOY_LABELS)) % len(texts)]} (ref {i})", label


def write_toy_classification_task(
    tasks_root: Path | str,
    base_model: str,
    *,
    task: str = "toy",
    n_train: int = 16,
    n_valid: int = 8,
    learning_rate: float = 5e-3,
    lora_rank: int = 4,
    lora_layers: int | str = "all",
    batch_size: int = 4,
    epochs: float = 2,
) -> Any:
    """Write a toy classification task spec and its curated data; return the loaded ``TaskSpec``.

    The spec goes to ``<tasks_root>/<task>/``; the curated ``{train,valid}.jsonl`` and their
    ``.meta.jsonl`` sidecars go to the workspace (``$TASKDISTILL_HOME/<task>/data/``), as curate writes them.
    """
    import json

    from taskdistill import paths
    from taskdistill.config import load_task

    task_dir = Path(tasks_root) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "labels.txt").write_text("\n".join(TOY_LABELS) + "\n", encoding="utf-8")
    (task_dir / "teacher_prompt.md").write_text("Classify the message.\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text(
        "\n".join(
            [
                f"task: {task}",
                "type: classification",
                "labels_file: labels.txt",
                "teacher:",
                "  base_url: http://127.0.0.1:9/v1",
                "  model: example/teacher-model",
                "  max_tokens: 16",
                "student:",
                f"  base_model: {json.dumps(str(base_model))}",
                f"  system_prompt: {json.dumps(TOY_SYSTEM_PROMPT)}",
                "  max_tokens: 8",
                "train:",
                "  profile: quick",
                f"  lora_rank: {lora_rank}",
                f"  lora_layers: {lora_layers}",
                f"  learning_rate: {learning_rate}",
                f"  batch_size: {batch_size}",
                f"  epochs: {epochs}",
                "  max_seq_len: 256",
                "  seed: 13",
                "cascade:",
                "  target: 0.9",
                "",
            ]
        ),
        encoding="utf-8",
    )
    data = paths.data_dir(task)
    splits = {"train": range(n_train), "valid": range(n_train, n_train + n_valid)}
    for split, indices in splits.items():
        rows, metas = [], []
        for i in indices:
            text, label = toy_example(i)
            messages = [
                {"role": "system", "content": TOY_SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": label},
            ]
            rows.append(json.dumps({"messages": messages}))
            metas.append(json.dumps({"input_hash": f"{i:064x}", "gold": label, "teacher": label, "meta": {}}))
        (data / f"{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (data / f"{split}.meta.jsonl").write_text("\n".join(metas) + "\n", encoding="utf-8")
    return load_task(str(task_dir))
