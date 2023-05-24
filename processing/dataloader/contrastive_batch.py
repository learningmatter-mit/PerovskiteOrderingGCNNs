import inspect
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.separate import separate
from torch_geometric.data.batch import Batch

from processing.dataloader.contrastive_collate import comp_collate

class CompBatch(Batch):

    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):

        batch, slice_dict, inc_dict = comp_collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], CompBatch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch
