#!/usr/bin/env python3
"""
Batch-query a local LLM and record questions, answers, and latency per question.
Requires:
    pip install pandas configparser llama-cpp-python
"""
from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import List

import pandas as pd
import configparser
from llama_cpp import Llama


# ──────────────────────────── utility functions ────────────────────────────────
def read_paragraphs(path: Path) -> List[str]:
    """Return non-empty paragraphs (separated by ≥1 blank line) from a text file."""
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]


def save_csv(
    questions: List[str],
    answers: List[str],
    times_ms: List[float],
    out_path: Path,
) -> None:
    """Write question/answer/timing triplets to a CSV file."""
    if not (len(questions) == len(answers) == len(times_ms)):
        raise ValueError("Input lists differ in length.")
    df = pd.DataFrame(
        {"Question": questions, "Answer": answers, "Time[ms]": times_ms}, index=range(1, len(questions) + 1)
    )
    df.to_csv(out_path, index_label="Index")


# ─────────────────────────────────── main ──────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = configparser.ConfigParser()
    cfg.read("parameters.ini")

    q_path = Path(cfg["paths"]["questions_path"]).expanduser()
    model_path = Path(cfg["paths"]["model_path"]).expanduser()
    out_csv = Path(f"QA_{model_path.stem}.csv")

    questions = read_paragraphs(q_path)
    logging.info("Loaded %d questions.", len(questions))

    llm = Llama(
        model_path=str(model_path),
        n_ctx=cfg.getint("llama", "n_ctx", fallback=2048),
        n_threads=cfg.getint("llama", "n_threads", fallback=6),
        n_gpu_layers=cfg.getint("llama", "n_gpu_layers", fallback=0),
    )

    answers: list[str] = []
    times_ms: list[float] = []

    for idx, prompt in enumerate(questions, 1):
        formatted = f"[INST] {prompt} [/INST]"
        t0 = time.perf_counter()
        try:
            out = llm(formatted, max_tokens=1024, temperature=0.0, top_p=1.0, top_k=1, stop=["</s>"])
            answer = out["choices"][0]["text"].strip()
        except Exception as exc:
            logging.error("Generation failed for Q%03d: %s", idx, exc)
            answer = "<ERROR>"
        dt_ms = (time.perf_counter() - t0) * 1000
        answers.append(answer)
        times_ms.append(dt_ms)

    save_csv(questions, answers, times_ms, out_csv)
    logging.info("Results written to %s", out_csv)


if __name__ == "__main__":
    main()

