# FusionSentPrototype

FusionSentPrototype is a Python script for analyzing public sentiment toward nuclear fusion in U.S.-focused news articles using a large language model (LLM). It performs multi-pass processing to filter, score, and summarize article content, generating both intermediate JSON output and a final sentiment report.

## Features

- Automatically scans `.log` files containing news articles.
- Filters for English-language articles primarily about the **United States**.
- Uses an LLM (e.g., `llama3.2:3b`) to:
  - Score sentiment toward **nuclear fusion** on a 1–5 Likert scale.
  - Justify each sentiment score in a single sentence.
  - Summarize sentiment trends across time-based batches.
  - Generate a final cohesive report on sentiment and temporal trends.

## Configuration

Edit the top of the script to match your environment:
```python
FOLDER = r"C:\Users\Cade\RealFusionProject"
MODEL = 'llama3.2:3b'
LLM_CTX_* = The designated context window you want to use. Refer to comments.

Under the prompt for summaries, temperature allows creativity with summarization. If you desire replicability, reduce the temperature. [0,1] range.

Optional 'batch' output code added, not thoroughly tested, review. This allows you to stop and begin the process, in theory. Code is generally commented at the designated blocks.
```

## Output

- `sentiment_report.txt`: Final human-readable report with batch summaries and overall insights.
- `all_sentiment_outputs.json`: Intermediate structured data including scores, dates, and justifications.

## Execution Overview

1. **Pass 1 – Article Analysis**
   - Reads all `.log` files in the directory.
   - Filters for U.S.-relevant, English articles.
   - Uses the LLM to assign sentiment scores and brief justifications.

2. **Pass 2 – Batch Summarization**
   - Groups relevant articles by batch (default: 100 articles).
   - Computes statistics and uses LLM to summarize each batch.

3. **Pass 3 – Final Summary**
   - Compiles a final report using statistics and prior summaries.

## Notes

- Articles not focused on the U.S. are marked with `"IRRELEVANT"` and excluded from scoring.
- You may adjust `BATCH_SIZE` and LLM context lengths (`LLM_CTX_*`) based on available model capacity.

## License

This script is for academic and research purposes only. Attribution required for derivative work.
