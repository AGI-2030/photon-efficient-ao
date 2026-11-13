"""Ring buffer of visited loop states, kept in host memory with the images in float16."""

from __future__ import annotations

import torch

from .environment import EpisodeBatch


class ReplayBuffer:
    def __init__(self, capacity: int, image_size: int, nmodes: int):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.previous = torch.empty((capacity, image_size, image_size), dtype=torch.float16)
        self.current = torch.empty_like(self.previous)
        self.delta_command = torch.empty((capacity, nmodes))
        self.log_exposure = torch.empty((capacity,))
        self.label = torch.empty((capacity, nmodes))

    def add(self, batch: EpisodeBatch) -> None:
        n = batch.label.shape[0]
        if n > self.capacity:
            raise ValueError("rollout larger than the buffer")
        index = (torch.arange(n) + self.cursor) % self.capacity
        self.previous[index] = batch.previous.cpu()
        self.current[index] = batch.current.cpu()
        self.delta_command[index] = batch.delta_command.cpu()
        self.log_exposure[index] = batch.log_exposure.cpu()
        self.label[index] = batch.label.cpu()
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def gather(self, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Samples at the given buffer positions (a CPU index tensor), images converted to float32."""
        return (
            self.previous[index].float(),
            self.current[index].float(),
            self.delta_command[index],
            self.log_exposure[index],
            self.label[index],
        )
