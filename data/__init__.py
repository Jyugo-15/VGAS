# """Dataset utilities for SmolVLA Q-Chunking."""

# try:  # pragma: no cover - optional dependency
#     from .datamodule import SmolVLAQChunkDataModule, TrajectoryBatch
# except ImportError:  # pragma: no cover
#     SmolVLAQChunkDataModule = None  # type: ignore[assignment]
#     TrajectoryBatch = None  # type: ignore[assignment]
# from .hflibero_loader import (
#     get_episode_metadata,
#     iter_episode_frames,
#     load_hflibero_dataset,
# )
# from .push_to_hub import push_suite_to_hub

# __all__ = ["load_hflibero_dataset", "get_episode_metadata", "iter_episode_frames", "push_suite_to_hub"]
# if SmolVLAQChunkDataModule is not None and TrajectoryBatch is not None:
#     __all__.extend(["SmolVLAQChunkDataModule", "TrajectoryBatch"])
