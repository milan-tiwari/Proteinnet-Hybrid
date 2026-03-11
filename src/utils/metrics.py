from __future__ import annotations

from typing import Tuple

import torch


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align P onto Q using the Kabsch algorithm.

    Args:
        P: (N, 3) predicted points
        Q: (N, 3) target points

    Returns:
        P_aligned: (N, 3) P after optimal rotation + translation
        R: (3, 3) rotation matrix
    """
    if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"Expected P and Q to be (N,3). Got P={tuple(P.shape)}, Q={tuple(Q.shape)}")

    # Center
    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    Pc = P - P_mean
    Qc = Q - Q_mean

    # Covariance
    C = Pc.T @ Qc  # (3,3)

    # SVD
    V, S, Wt = torch.linalg.svd(C, full_matrices=False)  # V:(3,3), Wt:(3,3)

    # Proper right-handed coordinate system: handle reflection
    d = torch.sign(torch.det(V @ Wt))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))

    R = V @ D @ Wt
    P_aligned = (Pc @ R) + Q_mean
    return P_aligned, R


def rmsd(P: torch.Tensor, Q: torch.Tensor, align: bool = True) -> torch.Tensor:
    """Root-mean-square deviation between two point clouds.

    Args:
        P: (N,3)
        Q: (N,3)
        align: if True, align P to Q with Kabsch before RMSD.

    Returns:
        Scalar tensor RMSD.
    """
    if align:
        P, _ = kabsch_align(P, Q)

    diff = P - Q
    return torch.sqrt(torch.mean(torch.sum(diff * diff, dim=-1)))


@torch.no_grad()
def batch_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    align: bool = True,
) -> torch.Tensor:
    """Compute mean RMSD across a padded batch.

    Args:
        pred: (B, L, 3)
        target: (B, L, 3)
        mask: (B, L) bool; valid points

    Returns:
        Scalar RMSD averaged across proteins that have >= 3 valid points.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.bool()

    B = pred.shape[0]
    rmsds = []
    for i in range(B):
        m = mask[i]
        if m.sum() < 3:
            continue
        Pi = pred[i][m]
        Qi = target[i][m]
        rmsds.append(rmsd(Pi, Qi, align=align))

    if not rmsds:
        return torch.tensor(float("nan"), device=pred.device)
    return torch.stack(rmsds).mean()
