import inspect
from torch import nn
from typing import (
    Tuple,
    Optional,
    Callable,
    Any,
    Dict,
    List
)
import torch


class DualOpt:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        moun_cls: Optional[Any] = None,
        moun_kwargs: Optional[Dict] = None,
        embed_cls: Optional[Any] = None,
        embed_kwargs: Optional[Dict] = None,
        adjust_lr_fn: Optional[Callable] = None,
    ):
        # normalize classes / kwargs
        moun_cls = moun_cls or getattr(torch.optim, "Muon", None)
        if moun_cls is None:
            raise ValueError("Muon optimizer class not found in torch.optim. Please provide moun_cls.")
        moun_kwargs = moun_kwargs or {"lr": lr, "weight_decay": weight_decay, "momentum": momentum, "nesterov": nesterov}
        embed_cls = embed_cls or getattr(torch.optim, "AdamW")
        embed_kwargs = embed_kwargs or {"lr": lr, "weight_decay": weight_decay}

        self.adjust_lr_fn = adjust_lr_fn
        self._step_counter = 0
        
        moun_params_names = set()
        adam_params_names_no_dacay = set()
        adam_params_names_decay = set()

        for n, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                adam_params_names_decay.add(f"{n}.weight")

            elif isinstance(mod, nn.Linear):
                if mod.bias is not None:
                  adam_params_names_no_dacay.add(f"{n}.bias")
                moun_params_names.add(f"{n}.weight")
            
            elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if mod.bias is not None:
                   adam_params_names_no_dacay.add(f"{n}.bias")
                adam_params_names_decay.add(f"{n}.weight")

            elif isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                adam_params_names_no_dacay.add(f"{n}.weight")
                if mod.bias is not None:
                   adam_params_names_no_dacay.add(f"{n}.bias")

        print(f"Found decay for adam: {len(adam_params_names_decay)}: {', '.join(list(adam_params_names_decay)[:15])}...\n")
        print(f"Found no decay for adam: {len(adam_params_names_no_dacay)}: {', '.join(list(adam_params_names_no_dacay)[:15])}...\n")
        print(f"Found moun params:{len(moun_params_names)}: {', '.join(list(moun_params_names)[:15])}...\n")

        moun_params = []
        adam_params_decay = []
        adam_params_no_decay = []

        for n, p in model.named_parameters():
            if n in moun_params_names:
                moun_params.append(p)
            elif n in adam_params_names_decay:
                adam_params_decay.append(p)

            elif n in adam_params_names_no_dacay:
                adam_params_no_decay.append(p)
        
        self.moun_param_num = len(moun_params)
        self.adam_param_num = len(adam_params_no_decay) + len(adam_params_decay)
        self.moun = moun_cls(moun_params, **moun_kwargs)
        if self.adam_param_num:
            fused = "fused" in inspect.signature(torch.optim.Adam).parameters
            optimizer_grouped_parameters = [
                {"params": adam_params_decay, "weight_decay": weight_decay},
                {"params": adam_params_no_decay, "weight_decay": 0.0},
            ]
            if fused:
              embed_kwargs.update({"fused": True})
            self.adam = embed_cls(optimizer_grouped_parameters, **embed_kwargs)
        
    def zero_grad(self, set_to_none: bool = False):
        self.moun.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None):
        if self.adjust_lr_fn is not None:
            try:
                self.adjust_lr_fn(self, self._step_counter)
            except Exception as e:
                # don't crash training for LR adjustment errors; surface a friendly message
                print(f"[DualOpt] adjust_lr_fn raised exception: {e}")

        self._step_counter += 1

        try:
            if self.moun_param_num >0:
               self.moun.step(closure=closure)
        except TypeError:
            self.moun.step()
        try:
            if self.adam_param_num > 0:
               self.adam.step(closure=closure)
        except TypeError:
            self.adam.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "moun": self.moun.state_dict(),
            "adam": self.adam.state_dict(),
            "step": self._step_counter
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "moun" in state:
            self.moun.load_state_dict(state["moun"])

        if "adam" in state:
            self.adam.load_state_dict(state["adam"])

        if "step" in state:
            self._step_counter = state["step"]

    def param_groups(self) -> Tuple[List[Dict], List[Dict]]:
        return self.moun.param_groups, self.adam.param_groups

    def add_param_group_to_moun(self, group):
        self.moun.add_param_group(group)

    def add_param_group_to_embed(self, group):
        self.adam.add_param_group(group)
