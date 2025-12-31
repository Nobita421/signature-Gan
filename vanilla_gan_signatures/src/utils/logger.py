"""GAN Training Logger - Tracks metrics, saves logs to CSV/JSON."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class GANLogger:
    """Logger for GAN training metrics and configuration."""

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """Initialize logger with output directory and experiment name."""
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._base_name = f"{experiment_name}_{self.timestamp}"
        print(f"[GANLogger] Initialized: {self._base_name}")

    def log_metrics(
        self, epoch: int, g_loss: float, d_loss: float, d_real: float, d_fake: float
    ) -> None:
        """Log training metrics for an epoch with console output."""
        entry = {
            "epoch": epoch,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "d_real": d_real,
            "d_fake": d_fake,
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics.append(entry)
        print(
            f"[Epoch {epoch:04d}] G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f} | "
            f"D(real): {d_real:.4f} | D(fake): {d_fake:.4f}"
        )

    def log_config(self, config_dict: dict[str, Any]) -> None:
        """Save hyperparameters configuration."""
        self.config = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            **config_dict,
        }
        print(f"[GANLogger] Config logged: {len(config_dict)} parameters")

    def save_to_csv(self) -> Path:
        """Save metrics to CSV file."""
        csv_path = self.log_dir / f"{self._base_name}_metrics.csv"
        if not self.metrics:
            print("[GANLogger] No metrics to save")
            return csv_path
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics)
        print(f"[GANLogger] Metrics saved: {csv_path}")
        return csv_path

    def save_to_json(self) -> Path:
        """Save metrics and config to JSON file."""
        json_path = self.log_dir / f"{self._base_name}_log.json"
        data = {"config": self.config, "metrics": self.metrics}
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[GANLogger] Full log saved: {json_path}")
        return json_path

    def get_summary(self) -> dict[str, Any]:
        """Return final training statistics."""
        if not self.metrics:
            return {"status": "no_data"}
        
        g_losses = [m["g_loss"] for m in self.metrics]
        d_losses = [m["d_loss"] for m in self.metrics]
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.metrics),
            "final_g_loss": self.metrics[-1]["g_loss"],
            "final_d_loss": self.metrics[-1]["d_loss"],
            "min_g_loss": min(g_losses),
            "min_d_loss": min(d_losses),
            "avg_g_loss": sum(g_losses) / len(g_losses),
            "avg_d_loss": sum(d_losses) / len(d_losses),
        }
        print(f"[GANLogger] Summary: {summary['total_epochs']} epochs completed")
        return summary
