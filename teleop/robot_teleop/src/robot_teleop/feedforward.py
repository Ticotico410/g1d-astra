from __future__ import annotations

from pathlib import Path

import numpy as np


class PinocchioRNEAFeedforward:
    """Gravity feedforward torque from Pinocchio RNEA for configured joints.

    Evaluated at zero velocity and acceleration, so the result is the gravity
    compensation term for the current posture.
    """

    def __init__(self, urdf_path: str, command_joints: list[str], output_joints: list[str]) -> None:
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(Path(urdf_path).expanduser().resolve()))
        self.data = self.model.createData()
        self.command_joints = list(command_joints)
        self.output_joints = list(output_joints)
        self.q_index: dict[str, int] = {}
        self.v_index: dict[str, int] = {}
        for name in set(self.command_joints + self.output_joints):
            jid = self.model.getJointId(name)
            if jid == 0 or jid >= self.model.njoints:
                raise ValueError(f"Joint name {name!r} not found in Pinocchio model")
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Only 1-DoF joints are supported for RNEA feedforward: {name}")
            self.q_index[name] = int(joint.idx_q)
            self.v_index[name] = int(joint.idx_v)

    def compute(self, joint_positions: dict[str, float]) -> dict[str, float]:
        q = self.pin.neutral(self.model)
        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)
        for name, value in joint_positions.items():
            idx = self.q_index.get(name)
            if idx is not None:
                q[idx] = float(value)
        tau = np.asarray(self.pin.rnea(self.model, self.data, q, v, a), dtype=float).reshape(-1)
        return {name: float(tau[self.v_index[name]]) for name in self.output_joints}
