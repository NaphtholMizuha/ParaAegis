from dp_accounting.dp_event import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

def find_dp_noise_multiplier(target_epsilon, delta, n_steps, sensitivity=1.0, tol=1e-6):
    left, right = 0.0, 10000.0
    while right - left > tol:
        mid = (left + right) / 2
        accountant = RdpAccountant()
        for _ in range(n_steps):
            accountant.compose(GaussianDpEvent(mid / sensitivity))
        total_epsilon = accountant.get_epsilon(delta)
        if total_epsilon > target_epsilon:
            left = mid
        else:
            right = mid
    return right  # 保证和 opacus 乘 CLIP_NORM 一致
