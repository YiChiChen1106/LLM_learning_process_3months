# Error Taxonomy for LoRA Adapter Evaluation

This file defines a small manual error taxonomy for evaluating LoRA / QLoRA adapter outputs.

The goal is not to produce a perfect automatic score. The goal is to make repeated evaluation less vague: when an answer is wrong, label what kind of wrong it is.

## Labels

### `factual_error`

The answer states something objectively wrong.

Examples:

- Says QLoRA quantizes the LoRA adapter.
- Says `adapter_model.safetensors` contains the full base model.
- Says LoRA updates all base model weights.

### `concept_confusion`

The answer mixes two related concepts.

Examples:

- Confuses LoRA with pretraining.
- Confuses LoRA with model compression.
- Confuses `attention_mask` with `labels=-100`.

### `hallucinated_mechanism`

The answer invents a mechanism, file, conversion path, or training process that was not part of the actual workflow.

Examples:

- Says adapter files must be converted to `base_model.ckpt`.
- Says LoRA removes unrelated base model parameters.
- Says adapter checkpoint is loaded as a standalone model.

### `incomplete_answer`

The answer is partly correct but misses a required key point.

Examples:

- Says adapter cannot generate alone, but does not say it needs the base model.
- Says QLoRA saves memory, but does not say the saved part is frozen base weights.
- Says `-100` ignores loss, but does not mention prompt or padding masking when the prompt asks for it.

### `format_issue`

The answer may be mostly correct, but does not follow the desired response format.

Examples:

- Does not start with a concise conclusion when the style expects it.
- Gives a long rambling answer when a short answer was requested.
- Uses unclear wording even if the concept is mostly correct.

### `off_topic`

The answer does not address the prompt.

Examples:

- The prompt asks about LoRA adapter checkpoints, but the answer talks about KV cache.
- The prompt asks whether QLoRA quantizes adapter weights, but the answer talks only about batch size.

## Suggested Manual Fields

Use these fields when annotating eval outputs:

```json
{
  "prompt_id": "adapter_only_rephrase",
  "run": "lora-v5",
  "is_correct": false,
  "error_types": ["incomplete_answer"],
  "note": "Says adapter cannot generate alone, but does not mention loading the base model."
}
```

`error_types` can contain multiple labels when needed.

## Minimal Passing Standard

For today's LoRA / QLoRA learning adapter, an answer is considered acceptable if:

- It gives the core factual answer.
- It does not introduce a contradictory mechanism.
- It does not confuse LoRA, QLoRA, adapter checkpoint, or full fine-tuning.
- It is short enough to avoid unsupported claims.

