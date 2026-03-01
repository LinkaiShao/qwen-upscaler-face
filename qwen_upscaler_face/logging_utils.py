"""Logging and memory monitoring utilities."""

import logging
import os
import sys

import psutil
import torch


def setup_logging(output_dir):
    """Configure dual file+stdout logging. Returns (logger, log_path)."""
    log_path = os.path.join(output_dir, "train.log")
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("train"), log_path


def mem_stats():
    """Return a string summarizing current RAM and GPU memory usage."""
    proc = psutil.Process()
    ram_gb = proc.memory_info().rss / 1e9
    parts = [f"RAM: {ram_gb:.1f}GB"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            parts.append(f"GPU{i}: alloc={alloc:.1f}GB res={reserved:.1f}GB")
    return " | ".join(parts)
