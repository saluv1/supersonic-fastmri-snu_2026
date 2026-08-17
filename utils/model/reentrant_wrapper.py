from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class ReentrantWrapper(nn.Module):

    DEFAULT_CHECKPOINT = True
    
    class _DummyWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
    
        def forward(self, *args):
            x, dummy = args[:-1], args[-1]
            return self.module(*x)
            
    def __init__(
        self, 
        module: nn.Module, 
        use_checkpoint: Optional[bool] = None
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint or self.DEFAULT_CHECKPOINT
        self._dummy_tensor = nn.Parameter(torch.ones(1, requires_grad=True))
        
        self.module = self._DummyWrapper(module)
    
    def forward(self, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(self.module, *args, self._dummy_tensor, use_reentrant=True)
        else:
            return self.module(*args, self._dummy_tensor)