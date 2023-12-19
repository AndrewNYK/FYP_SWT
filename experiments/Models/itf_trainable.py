# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from typing import Optional, Callable, Union, List

class Itrainable:    
    def train_epoch(self,
              loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              val_loader: Optional[torch.utils.data.DataLoader],
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
              device: Optional[torch.device],
              scaler: Optional[torch.cuda.amp.GradScaler],
              ):
        """Defines the interaction between the model and the dataloader
        (dataset) during training. The specific implementation depends on the model.
        Trains the model over the dataloader for 1 epoch and updates weights.
        """
        pass
    
    def val(self,
            loader: torch.utils.data.DataLoader,
            loss_fn: Union[Callable[[Tensor,Tensor], Tensor],\
                           List[ Callable[[Tensor,Tensor], Tensor] ] ],
            device: Optional[torch.device]):
        pass