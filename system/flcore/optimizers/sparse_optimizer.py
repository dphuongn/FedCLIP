from transformers import AdamW 
from torch.optim import Optimizer
import torch
import math
import numpy as np

class SparseAdamW(AdamW):
    def __init__(self,
                sparse_lambda=0.1,  # Sparsity regularization coefficient
                lambda_schedule=None,  # Schedule for increasing lambda
                max_lambda=None,  # Maximum lambda value
                lambda_num=None,  # Number of lambda steps
                # clip_value=1.0,  # Clipping value for gradient
                **kwargs
                ):
        super().__init__(**kwargs)
        self.sparse_lambda = sparse_lambda
        print(f"Initial sparse lambda in optimizer: {self.sparse_lambda}")
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        # self.clip_value = clip_value  # Clipping gradients for stability
        self._build_lambda_list(max_lambda, lambda_num)
    
    def _build_lambda_list(self, max_lambda, lambda_num):
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
        elif self.lambda_schedule == "linear":
            assert max_lambda is not None and lambda_num is not None, "Specify max_lambda and lambda_num for linear schedule"
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            assert max_lambda is not None and lambda_num is not None, "Specify max_lambda and lambda_num for log-linear schedule"
            self._lambdas = np.logspace(np.log10(self.sparse_lambda), np.log10(max_lambda), lambda_num)
        elif self.lambda_schedule == "exp_linear":
            assert max_lambda is not None and lambda_num is not None, "Specify max_lambda and lambda_num for exp-linear schedule"
            self._lambdas = np.exp(np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num))
        else:
            raise NotImplementedError("Lambda schedule not supported.")
    
    def step_lambda(self):
        """Increase the sparsity threshold (lambda) according to the schedule."""
        if self._lambdas is None:
            print("No lambda schedule specified. Sparsity remains constant.")
            return
        else:
            if self.lambda_idx < len(self._lambdas) - 1:
                self.lambda_idx += 1
                self.sparse_lambda = self._lambdas[self.lambda_idx]
                print(f"Updated sparse lambda to {self.sparse_lambda}")
            else:
                print(f"Reached end of lambda schedule. Using sparse lambda = {self.sparse_lambda}")
    
    def step(self, closure=None):
        """
        Performs a single optimization step with sparsity enforcement.
        This includes applying the proximal operator for \ell_1-regularization.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Clip gradients to avoid explosion
        # torch.nn.utils.clip_grad_norm_(self.param_groups[0]["params"], self.clip_value)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported by AdamW")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Apply AdamW weight decay
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

                # Apply gradient descent
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Sparsity enforcement (Proximal operator for \ell_1 regularization)
                if self.sparse_lambda > 0:
                    p.data[p.data > self.sparse_lambda] -= self.sparse_lambda
                    p.data[p.data < -self.sparse_lambda] += self.sparse_lambda
                    p.data[abs(p.data) < self.sparse_lambda] = 0.0  # Pruning step

        return loss
