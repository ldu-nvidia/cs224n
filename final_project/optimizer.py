from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdamW(Optimizer):
    # params: named or unnamed tensor or dicts, specifies what tensors should be optimized
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # a dict containing default values of optimization options
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None

        #LBFGS need to reevaluate the function multiple times so have to pass in a closure that allows them to recompute your model, 
        # should clear the gradients, compute the loss and return it
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #print("group: ", group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]
                ## need to put these states on device to trigger GPU compute
                if "first_momentum" not in state:
                    state["first_momentum"] = torch.zeros(p.grad.shape, dtype=torch.float32).to(device)
                if "second_momentum" not in state:
                    state["second_momentum"] = torch.zeros(p.grad.shape, dtype=torch.float32).to(device)
                if "time_step" not in state:
                    state["time_step"] = torch.tensor(0, dtype=torch.int).to(device)

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE

                # update biased first momentum estimate
                # update time step
                state["time_step"] += 1
                state["first_momentum"] = group["betas"][0]*state["first_momentum"] + (1.0 - group["betas"][0])*grad
                
                # update biased second momentum estimate
                state["second_momentum"] = group["betas"][1]*state["second_momentum"] + (1.0 - torch.tensor(group["betas"][1]))*torch.pow(grad, 2)

                # adaptive learning rate
                alpha_t = alpha * math.sqrt(1.0 - pow(group["betas"][1], state["time_step"]))/(1.0 - pow(group["betas"][0], state["time_step"]))
                # update certain parameter by adaptive gradient, first momentum smooth the DIRECTION of gradient, prevent noisy gradient direction
                # second momentum scale the MAGNITUDE of gradient, accelerate direction of small gradient magnitude and decelerate direction of large gradient
                p.data -= alpha_t * state["first_momentum"]/(state["second_momentum"].sqrt() + group["eps"])
                # coefficient should be alpha instead of alpha_t for parameter regularization
                p.data -= alpha * group["weight_decay"] * p.data
        return loss
