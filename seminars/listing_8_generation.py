"""
Generative pipeline listing.
"""

# pylint: disable=duplicate-code, too-many-locals, unused-argument

import time
from pathlib import Path

try:
    from pandas import read_csv
except ImportError:
    print('Library "pandas" not installed. Failed to import.')

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print('Library "torch" not installed. Failed to import.')

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')


class TextDataset(Dataset):
    """
    TaskDataset returning a single text column.
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> str:
        return self._texts[index]


def inspect_generative_tokenizers() -> None:
    """
    Demonstrates differences in padding configuration for generative models.
    """

    gpt_tok = AutoTokenizer.from_pretrained("gpt2")

    # 1. Missing pad token in causal models
    print(f"GPT-2 pad_token by default: {gpt_tok.pad_token}")
    print("We set `gpt_tok.pad_token = gpt_tok.eos_token` to enable batching.\n")

    # 2. Padding side importance
    print(f"GPT-2 default padding_side: '{gpt_tok.padding_side}'")
    print("Why we set: gpt_tok.padding_side = 'left' -> Decoder-only models generate")
    print("to the right, so padding zeros must be placed on the left!\n")


def run_seq2seq(texts: list[str], batch_size: int = 4) -> None:
    """
    Runs batched translation.
    """

    model_name = "Helsinki-NLP/opus-mt-en-ru"

    # 3. Load tokenizer and seq2seq generative model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

    # 4. Prepare dataloader
    loader = DataLoader(TextDataset(texts), batch_size=batch_size, shuffle=False)
    print(f"Total texts: {len(texts)} | Number of batches: {len(loader)}\n")

    # 5. Get translations
    start_time = time.time()
    results = []

    with torch.no_grad():
        for batch_idx, batch_texts in enumerate(loader):
            tokens = tokenizer(
                list(batch_texts), padding=True, truncation=True, return_tensors="pt"
            )
            output_ids = model.generate(**tokens, max_new_tokens=25)

            # No slicing needed for seq2seq task. Output contains only target translation
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)
            print(
                (f"\nProcessed batch {batch_idx + 1}/{len(loader)}\n"),
                (f"tokens shape: {tokens['input_ids'].shape})\n"),
            )

    print(f"Finished in {time.time() - start_time:.2f}s\n")

    # 6. Print results
    for orig, trans in zip(texts[:2], results[:2]):
        print(f"EN: '{orig}'\n RU: '{trans}'\n")


def run_causal(texts: list[str], batch_size: int = 4) -> None:
    """
    Runs batched causal generation.
    """

    model_name = "gpt2"

    # 7. Load tokenizer and causal generative model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Left padding is required for Causal LM batching
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 8. Prepare dataloader
    loader = DataLoader(TextDataset(texts), batch_size=batch_size, shuffle=False)
    batch_prompts = list(next(iter(loader)))

    # 9. Tokenize
    tokens = tokenizer(batch_prompts, padding=True, return_tensors="pt")
    prompt_len = tokens["input_ids"].shape[1]

    # 10. Run inference
    with torch.no_grad():
        output_ids = model.generate(
            **tokens,
            max_new_tokens=15,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 11. Apply post-processing: slice off prompt tokens
    continuations = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)

    # 12. Print results
    for prompt, continuation in zip(batch_prompts[:2], continuations[:2]):
        print(f"\nPrompt: '{prompt}' -> Continuation: '{continuation.strip()}'\n")


def main() -> None:
    """
    Entrypoint for the listing.
    """

    # 13. Inspect tokenizer
    inspect_generative_tokenizers()

    data_path = Path(__file__).parent / "assets" / "glue_mnli_example.csv"

    # 14. Take first 12 real English sentences from the CSV
    sample_texts = read_csv(data_path)["premise"].head(12).tolist()

    # 15. Run Seq2Seq translation on batches
    run_seq2seq(sample_texts, batch_size=4)

    # 16. Run Causal LM generation on batches
    run_causal(sample_texts, batch_size=4)


# TASKS


# Task 1:
def configure_causal_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    """
    Configure a Causal LM tokenizer for batching by assigning eos_token as pad_token
    and setting padding_side to 'left'.

    Args:
        tokenizer (AutoTokenizer): Hugging Face tokenizer instance.

    Returns:
        AutoTokenizer: The configured tokenizer.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# tok = configure_causal_tokenizer(AutoTokenizer.from_pretrained("gpt2"))
# (tok.pad_token == tok.eos_token, tok.padding_side) -> (True, 'left')


# Task 2:
def get_prompt_token_length(
    prompt: str,
    tokenizer: AutoTokenizer,
) -> int:
    """
    Tokenize a single prompt and return its token length.

    Args:
        prompt (str): Raw input prompt string.
        tokenizer (AutoTokenizer): Hugging Face tokenizer.

    Returns:
        int: Number of tokens in the prompt.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# get_prompt_token_length("Hello world", tokenizer) -> 2


# Task 3:
def slice_prompt_tokens(
    generated_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Slice off the input prompt tokens from a causal LM generation output tensor.

    Args:
        generated_ids (torch.Tensor): Output tensor of shape [batch_size, total_seq_len].
        prompt_len (int): Length of the prompt in tokens.

    Returns:
        torch.Tensor: Tensor containing only newly generated tokens.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# slice_prompt_tokens(torch.tensor([[10, 20, 30, 40]]), 2) -> tensor([[30, 40]])
# slice_prompt_tokens(torch.tensor([[1, 2, 3], [4, 5, 6]]), 1) -> tensor([[2, 3], [5, 6]])


# Task 4:
def decode_and_clean_batch(
    token_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """
    Decode a batch of token IDs into strings, skipping special tokens,
    and stripping leading/trailing whitespace from each result.

    Args:
        token_ids (torch.Tensor): Tensor of token IDs of shape [batch_size, seq_len].
        tokenizer (AutoTokenizer): Hugging Face tokenizer instance.

    Returns:
        list[str]: Cleaned list of generated strings.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# decode_and_clean_batch(torch.tensor([[15496, 11]]), tokenizer) -> ['Hello,']


# Task 5:
def truncate_at_first_newline(generated_texts: list[str]) -> list[str]:
    """
    Post-process generated texts by keeping only the text before the first newline character.
    If no newline is present in a string, keep the entire string.

    Args:
        generated_texts (list[str]): List of raw generated text strings.

    Returns:
        list[str]: Truncated text strings.
    """
    # student implementation goes here
    raise NotImplementedError


# Function calls with expected result:
# truncate_at_first_newline(["First line\nSecond line", \
# "Single line\n"]) -> ['First line', 'Single line']
# truncate_at_first_newline(["\nEmpty start", "No newline"]) -> ['', 'No newline']


if __name__ == "__main__":
    main()
