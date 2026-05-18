import torch
from typing import Optional, Union, List, Dict


class HiddenStatesEditingMethod:
    def __init__(self, 
        if_edit: bool=False,
        edit_layer_idx: Optional[List[int]]=None, 
        edit_mask: Optional[torch.BoolTensor]=None, # (seq_len)
        edit_tensor: Optional[Dict[int, torch.FloatTensor]]=None, # dict(layer_idx: (bs=1, seq_len, hidden_size))
    ):
        assert if_edit == False or (\
            edit_layer_idx is not None and \
            edit_mask is not None and \
            edit_tensor is not None), "if_edit is True, all edit arguments should be provided."
        
        if if_edit:
            assert all(edit_mask.shape[0] == edit_tensor[layer_idx].shape[1] for layer_idx in edit_layer_idx), "edit_mask and edit_tensor should have the same input length."
            # edit_mask must be bool
            assert edit_mask.dtype == torch.bool, "edit_mask should be torch.bool."
        
        self.if_edit = if_edit
        self.edit_layer_idx = edit_layer_idx
        self.edit_mask = edit_mask
        self.edit_tensor = edit_tensor
        self.saved_hs = {}
        for layer_idx in self.edit_layer_idx:
            self.saved_hs[layer_idx] = []
    
    def edit(self, hidden_states, layer_idx):
        # masked = False : to func(hs, edit)
        # masked = True : keep
        edit_val = self.edit_tensor[layer_idx]
        assert edit_val.shape == hidden_states.shape
        edit_val = edit_val.to(hidden_states.device)
        self.edit_mask = self.edit_mask.to(hidden_states.device)
        tmp = (hidden_states + edit_val) * (~self.edit_mask).unsqueeze(-1) 
        new_hidden_states = hidden_states * self.edit_mask.unsqueeze(-1) + tmp
        return new_hidden_states