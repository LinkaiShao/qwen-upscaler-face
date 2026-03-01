"""Training plot utilities for Qwen upscaler."""

import logging
import os


def save_training_plots(step_hist, flow_hist, id_hist, val_steps, val_mse, output_dir):
    """Save training loss + validation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Flow loss
        axes[0].plot(step_hist, flow_hist, alpha=0.6, linewidth=0.5)
        if len(step_hist) > 50:
            window = min(50, len(flow_hist))
            smoothed = [sum(flow_hist[max(0, i - window):i + 1]) / len(flow_hist[max(0, i - window):i + 1])
                        for i in range(len(flow_hist))]
            axes[0].plot(step_hist, smoothed, color="red", linewidth=1.5, label="smoothed")
            axes[0].legend()
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Face-Weighted Flow Loss")
        axes[0].set_title("Flow Loss")
        axes[0].grid(True, alpha=0.3)

        # Identity loss
        axes[1].plot(step_hist, id_hist, alpha=0.6, linewidth=0.5, color="orange")
        if len(step_hist) > 50:
            window = min(50, len(id_hist))
            smoothed_id = [sum(id_hist[max(0, i - window):i + 1]) / len(id_hist[max(0, i - window):i + 1])
                           for i in range(len(id_hist))]
            axes[1].plot(step_hist, smoothed_id, color="red", linewidth=1.5, label="smoothed")
            axes[1].legend()
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Identity Loss (1 - cos_sim)")
        axes[1].set_title("Face Identity Loss")
        axes[1].grid(True, alpha=0.3)

        # Validation MSE
        if val_steps:
            axes[2].plot(val_steps, val_mse, "o-", color="green")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Val Latent MSE")
        axes[2].set_title("Validation MSE")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_plots.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        logging.getLogger("train").warning(f"Failed to save plots: {e}")
