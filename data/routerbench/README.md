# RouterBench Dataset

This directory holds the RouterBench dataset files used by
`eval/routerbench.py`.

## Expected file

Place the dataset here as:

```
data/routerbench/routerbench.jsonl
```

## Format

The file must be a JSONL (JSON Lines) file where each line is a JSON
object with at least a `"prompt"` key:

```json
{"prompt": "What is the capital of France?", "label": "Paris", "metadata": {"category": "geography"}}
{"prompt": "Solve: 2 + 2 = ?", "label": "4", "metadata": {"category": "math"}}
```

### Required keys

| Key      | Type   | Description                        |
|----------|--------|------------------------------------|
| `prompt` | string | The input prompt for the model     |

### Optional keys

| Key               | Type   | Description                              |
|-------------------|--------|------------------------------------------|
| `label`           | string | Ground-truth answer for accuracy scoring |
| `expected_answer` | string | Alternative key for the ground truth     |
| `metadata`        | object | Arbitrary metadata (preserved in output) |

## Downloading the data

RouterBench is an academic benchmark.  To obtain the dataset:

1. Visit the RouterBench project page or paper repository.
2. Download the dataset following the authors' instructions.
3. Convert the data to the JSONL format described above if necessary.
4. Place the resulting file at `data/routerbench/routerbench.jsonl`.

Once the file is in place, set `has_labels: true` in
`configs/eval.yaml → benchmarks → routerbench` if your file includes
ground-truth labels.

## Configuration

In `configs/eval.yaml`:

```yaml
benchmarks:
  routerbench:
    enabled: true
    dataset_path: "data/routerbench/routerbench.jsonl"
    has_labels: false   # set to true if labels are present
```

## Notes

- The dataset is NOT bundled with this repository.
- Do NOT commit large dataset files to git.  Add them to `.gitignore` or
  symlink from scratch storage.
