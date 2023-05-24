from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.typing import SparseTensor

def comp_collate(
    cls,
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
):
    data_list = [data for compdata in data_list for data in compdata.structures]
    return collate(cls,data_list,increment,add_batch,follow_batch,exclude_keys)

