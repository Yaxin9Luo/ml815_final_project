"""
Directly tokenize raw SlimPajama .jsonl.zst files per domain.

Streams chunk files one by one, tokenizes with GPT-2 BPE (tiktoken),
and stops once the target token count per domain is reached.
Outputs uint16 binary files ready for training data mixing.

Usage:
    python scripts/tokenize_slimpajama_direct.py
"""

import os
import json
import numpy as np
import tiktoken
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
RAW_DATA_DIR = '/data/yaxin/data/SlimPajama-627B-DC/train'
OUTPUT_DIR = '/data/yaxin/pretrain_data_analysis/data/slimpajama_tokenized'

# Token budget per domain: max proportion across all 3 experiments × 3B + 5M validation buffer
# Mix 1 (Balanced):   CC=20%, C4=20%, Book=20%, Wiki=10%, ArXiv=10%, Github=10%, SE=10%
# Mix 2 (Book-heavy): CC=5%,  C4=5%,  Book=70%, Wiki=10%, ArXiv=5%,  Github=3%,  SE=2%
# Mix 3 (Web-heavy):  CC=45%, C4=30%, Book=2%,  Wiki=7%,  ArXiv=3%,  Github=5%,  SE=8%
TRAIN_BUDGET = 3_000_000_000
VAL_BUFFER = 5_000_000

DOMAIN_TOKEN_TARGETS = {
    'RedPajamaCommonCrawl':   int(0.45 * TRAIN_BUDGET + VAL_BUFFER),  # 1.355B
    'RedPajamaC4':            int(0.30 * TRAIN_BUDGET + VAL_BUFFER),  # 0.905B
    'RedPajamaBook':          int(0.70 * TRAIN_BUDGET + VAL_BUFFER),  # 2.105B
    'RedPajamaWikipedia':     int(0.10 * TRAIN_BUDGET + VAL_BUFFER),  # 0.305B
    'RedPajamaArXiv':         int(0.10 * TRAIN_BUDGET + VAL_BUFFER),  # 0.305B
    'RedPajamaGithub':        int(0.10 * TRAIN_BUDGET + VAL_BUFFER),  # 0.305B
    'RedPajamaStackExchange': int(0.10 * TRAIN_BUDGET + VAL_BUFFER),  # 0.305B
}


def stream_jsonl_zst(file_path):
    """Yield JSON objects from a .jsonl.zst file."""
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = reader.read1  # low-level; use TextIOWrapper instead
            import io
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_reader:
                line = line.strip()
                if line:
                    yield json.loads(line)


def tokenize_domain(domain, target_tokens, enc, raw_dir, output_dir):
    """Tokenize a single domain until target token count is reached."""
    domain_path = Path(raw_dir) / domain
    out_path = Path(output_dir) / domain
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / 'train.bin'

    # Get sorted chunk files for deterministic ordering
    chunk_files = sorted(domain_path.glob('chunk_*.jsonl.zst'),
                         key=lambda f: int(f.stem.split('_')[1]))

    if not chunk_files:
        logging.warning(f"No chunk files found for {domain} at {domain_path}")
        return 0

    logging.info(f"[{domain}] Found {len(chunk_files)} chunk files. Target: {target_tokens:,} tokens")

    # Accumulate tokens in batches to manage memory
    all_token_arrays = []
    total_tokens = 0
    docs_processed = 0
    start_time = time.time()

    for chunk_file in chunk_files:
        if total_tokens >= target_tokens:
            break

        logging.info(f"[{domain}] Processing {chunk_file.name}... ({total_tokens:,}/{target_tokens:,} tokens)")

        for doc in stream_jsonl_zst(chunk_file):
            text = doc.get('text', '')
            if not text:
                continue

            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            all_token_arrays.append(np.array(ids, dtype=np.uint16))
            total_tokens += len(ids)
            docs_processed += 1

            if total_tokens >= target_tokens:
                break

        # Periodically log progress
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / max(elapsed, 1)
        logging.info(f"[{domain}] {total_tokens:,}/{target_tokens:,} tokens "
                     f"({docs_processed:,} docs, {tokens_per_sec:,.0f} tok/s)")

    # Concatenate and trim to target
    logging.info(f"[{domain}] Concatenating {len(all_token_arrays)} arrays...")
    arr = np.concatenate(all_token_arrays)
    if len(arr) > target_tokens:
        arr = arr[:target_tokens]

    # Write to binary file
    logging.info(f"[{domain}] Writing {len(arr):,} tokens to {out_file}")
    arr.tofile(out_file)

    elapsed = time.time() - start_time
    logging.info(f"[{domain}] Done! {len(arr):,} tokens, {docs_processed:,} docs, {elapsed:.1f}s")
    return len(arr)


def main():
    logging.info("=== SlimPajama Direct Tokenization ===")
    logging.info(f"Raw data: {RAW_DATA_DIR}")
    logging.info(f"Output: {OUTPUT_DIR}")

    enc = tiktoken.get_encoding("gpt2")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    token_counts = {}
    for domain, target in DOMAIN_TOKEN_TARGETS.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain} (target: {target:,} tokens)")
        logging.info(f"{'='*60}")
        count = tokenize_domain(domain, target, enc, RAW_DATA_DIR, OUTPUT_DIR)
        token_counts[domain] = count

    # Summary
    logging.info("\n" + "="*60)
    logging.info("TOKEN COUNTS SUMMARY")
    logging.info("="*60)
    for domain, count in token_counts.items():
        target = DOMAIN_TOKEN_TARGETS[domain]
        pct = count / target * 100
        logging.info(f"  {domain:30s}: {count:>15,} / {target:>15,} ({pct:.1f}%)")
    logging.info(f"  {'TOTAL':30s}: {sum(token_counts.values()):>15,}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
