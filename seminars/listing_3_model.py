"""
HuggingFace model listing.
"""

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


def main() -> None:
    """
    Entrypoint for the listing.
    """

    #########################
    # Classification scenario
    #########################

    # 1. Load tokenizer for classification model
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 2. Convert text to tokens
    text = "KFC заработал в Нижнем под новым брендом"
    tokens = tokenizer(text, return_tensors="pt")

    # 3. Import model
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/russian_toxicity_classifier")

    # 4. Classify text
    with torch.no_grad():
        output = model(**tokens)

    # 5. Get label
    predictions = torch.argmax(output.logits).item()

    # 6. Print predictions
    print(predictions)

    # 7. Map with labels
    labels = model.config.id2label
    print(labels[predictions])

    #########################
    # Generation scenario
    #########################

    # 8. Load tokenizer for generation model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 9. Convert text to tokens
    text = "Generate any text"
    tokens = tokenizer(text, return_tensors="pt")

    # 10. Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 11. Generate text
    output = model.generate(**tokens, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(results[0])


if __name__ == "__main__":
    main()
