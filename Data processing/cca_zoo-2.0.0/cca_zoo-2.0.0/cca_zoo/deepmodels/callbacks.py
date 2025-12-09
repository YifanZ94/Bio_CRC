from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader


class CorrelationCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # Handle both single DataLoader and list of DataLoaders
        val_dataloader = trainer.val_dataloaders
        if isinstance(val_dataloader, list) and len(val_dataloader) > 0:
            val_dataloader = val_dataloader[0]
        elif not isinstance(val_dataloader, DataLoader):
            return  # No validation dataloader available
        
        pl_module.log(
            "val/corr",
            pl_module.score(val_dataloader).sum(),
        )
