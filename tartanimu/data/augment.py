"""Rotation-equivariance augmentation for IMU data.

Applies random SO(3) rotations to both accelerometer and gyroscope
readings, preserving the physical relationship between them.
"""

import torch
import torch.nn as nn
import math


class RotationAugmentation(nn.Module):
    """Apply random 3D rotation to IMU data for augmentation.

    Rotates both acc and gyro channels by the same random rotation,
    which is a physically valid transformation (equivalent to mounting
    the IMU in a different orientation).
    """

    def __init__(self, max_angle: float = math.pi):
        super().__init__()
        self.max_angle = max_angle

    @staticmethod
    def random_rotation_matrix(max_angle: float, device: torch.device) -> torch.Tensor:
        """Generate a random rotation matrix via axis-angle."""
        # Random axis
        axis = torch.randn(3, device=device)
        axis = axis / axis.norm()

        # Random angle
        angle = torch.empty(1, device=device).uniform_(-max_angle, max_angle)

        # Rodrigues' formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ], device=device)

        R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def forward(
        self,
        imu: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            imu: [B, seq_len, 6, T] or [B, 6, T] IMU data
            velocity: [B, 3] optional velocity labels to rotate consistently
        Returns:
            rotated_imu, rotated_velocity (or None)
        """
        R = self.random_rotation_matrix(self.max_angle, imu.device)  # [3, 3]

        if imu.dim() == 4:
            B, S, C, T = imu.shape
            acc = imu[:, :, :3, :]   # [B, S, 3, T]
            gyro = imu[:, :, 3:, :]  # [B, S, 3, T]
            # R @ each 3D vector across time
            acc_rot = torch.einsum("ij,bsjt->bsit", R, acc)
            gyro_rot = torch.einsum("ij,bsjt->bsit", R, gyro)
            imu_rot = torch.cat([acc_rot, gyro_rot], dim=2)
        elif imu.dim() == 3:
            B, C, T = imu.shape
            acc = imu[:, :3, :]
            gyro = imu[:, 3:, :]
            acc_rot = torch.einsum("ij,bjt->bit", R, acc)
            gyro_rot = torch.einsum("ij,bjt->bit", R, gyro)
            imu_rot = torch.cat([acc_rot, gyro_rot], dim=1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {imu.dim()}D")

        vel_rot = None
        if velocity is not None:
            vel_rot = torch.einsum("ij,bj->bi", R, velocity)

        return imu_rot, vel_rot
