import pytorch_lightning as pl
import torch


class MemoryLogger(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e6

            pl_module.log('mem_usage_MB', mem)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
