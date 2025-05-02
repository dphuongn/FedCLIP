import math
import numpy as np
import torch
from torch.optim import AdamW

class SparseAdamW(AdamW):
    """
    AdamW optimizer with built-in L1 sparsity (proximal) regularization.

    Args:
        sparse_lambda (float): initial L1-regularization coefficient.
        correct_bias (bool): whether to apply Adam bias correction.
        lambda_schedule (Optional[Union[list, str]]): if a list, must be a list of lambda values;
            if "linear", "log_linear", or "exp_linear", will build a schedule between sparse_lambda
            and max_lambda using lambda_num steps.
        max_lambda (float): highest lambda value (required if schedule is string).
        lambda_num (int): number of steps in the schedule (required if schedule is string).
        **kwargs: passed through to torch.optim.AdamW (e.g., lr, betas, eps, weight_decay).
    """
    def __init__(
        self,
        *,
        sparse_lambda: float = 0.1,
        correct_bias: bool = True,
        lambda_schedule=None,
        max_lambda=None,
        lambda_num=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # core sparsity parameter
        self.sparse_lambda = sparse_lambda
        # build optional schedule
        self.lambda_schedule = lambda_schedule
        self.lambda_idx = 0
        self._build_lambda_list(max_lambda, lambda_num)

        # ensure every param_group has correct_bias
        for g in self.param_groups:
            g.setdefault("correct_bias", correct_bias)

    def _build_lambda_list(self, max_lambda, lambda_num):
        if self.lambda_schedule is None:
            self._lambdas = None
            return

        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
        elif self.lambda_schedule == "linear":
            assert max_lambda is not None and lambda_num is not None, \
                "Specify max_lambda and lambda_num for linear schedule"
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            assert max_lambda is not None and lambda_num is not None, \
                "Specify max_lambda and lambda_num for log-linear schedule"
            self._lambdas = np.logspace(
                np.log10(self.sparse_lambda),
                np.log10(max_lambda),
                lambda_num
            )
        elif self.lambda_schedule == "exp_linear":
            assert max_lambda is not None and lambda_num is not None, \
                "Specify max_lambda and lambda_num for exp-linear schedule"
            self._lambdas = np.exp(
                np.linspace(
                    np.log(self.sparse_lambda),
                    np.log(max_lambda),
                    lambda_num
                )
            )
        else:
            raise NotImplementedError(f"Lambda schedule '{self.lambda_schedule}' not supported.")

    def step_lambda(self):
        """Advance to the next lambda in the schedule."""
        if self._lambdas is None:
            print("No lambda schedule specified; sparse_lambda remains constant.")
            return

        if self.lambda_idx < len(self._lambdas) - 1:
            self.lambda_idx += 1
            self.sparse_lambda = float(self._lambdas[self.lambda_idx])
            print(f"Updated sparse_lambda -> {self.sparse_lambda}")
        else:
            print("Reached end of lambda schedule; no further update.")

    def step(self, closure=None):
        """
        Performs a single optimization step, then applies an L1-proximal operator
        to encourage sparsity.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported by SparseAdamW")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Weight decay (AdamW style)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

                # Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Proximal L1 shrinkage for sparsity
                lam = self.sparse_lambda
                if lam > 0:
                    # shrink toward zero
                    p.data = torch.where(p.data > lam, p.data - lam, p.data)
                    p.data = torch.where(p.data < -lam, p.data + lam, p.data)
                    # zero-out small entries
                    p.data[torch.abs(p.data) < lam] = 0.0

        return loss
