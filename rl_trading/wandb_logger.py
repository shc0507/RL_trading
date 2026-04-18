"""Thin wandb wrapper. No-op when disabled or wandb is not installed."""

from __future__ import annotations

from typing import Any

try:
    import wandb as _wandb  # type: ignore[import-not-found]
except ImportError:
    _wandb = None


class WandbLogger:
    def __init__(self, run: Any):
        self._run = run
        self._defined: set[str] = set()

    @classmethod
    def init(
        cls,
        enabled: bool,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        tags: list[str] | None,
        config: dict,
    ) -> "WandbLogger | None":
        if not enabled:
            return None
        if _wandb is None:
            raise ImportError(
                "wandb is not installed. Install with: uv pip install -e '.[wandb]'"
            )
        run = _wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            config=config,
            mode="online",
        )
        return cls(run)

    def define_context(self, ctx: str) -> None:
        """Set up a per-context epoch axis so each (agent/class/fold) has its own x-axis."""
        if ctx in self._defined:
            return
        self._run.define_metric(f"{ctx}/epoch")
        self._run.define_metric(f"{ctx}/*", step_metric=f"{ctx}/epoch")
        self._defined.add(ctx)

    def log(self, metrics: dict[str, float], ctx: str, epoch: int) -> None:
        payload = {f"{ctx}/{k}": v for k, v in metrics.items()}
        payload[f"{ctx}/epoch"] = epoch
        self._run.log(payload)

    def log_image(self, key: str, fig) -> None:
        assert _wandb is not None  # guaranteed if a logger instance exists
        self._run.log({key: _wandb.Image(fig)})

    def log_table(self, key: str, df) -> None:
        assert _wandb is not None
        self._run.log({key: _wandb.Table(dataframe=df)})

    def finish(self) -> None:
        self._run.finish()
