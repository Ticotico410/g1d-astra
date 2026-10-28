"""MuJoCo visualization of the commanded robot pose.

This is a viewer only: no physics stepping and no inverse kinematics. IK is
solved by `placo_robot.PlacoIKSolver`; this module just mirrors the resulting
joint angles so the operator can see what is being sent to the robot.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .math_utils import clamp_to_limits


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    resolved = (urdf_path.parent / path).resolve()
    if resolved.exists():
        return resolved
    fallback = (urdf_path.parent / "meshes" / path.name).resolve()
    if fallback.exists():
        return fallback
    return resolved


class MujocoVisualizer:
    """Load a URDF into MuJoCo and drive its joints from commanded angles.

    When `floating_base` is set, a free joint is injected at
    `floating_base_child_link` so a wheeled base can be integrated from velocity
    commands. Fixed-base models should set it to False.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        controlled_joints: list[str],
        floating_base: bool = False,
        floating_base_child_link: str | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.floating_base = floating_base
        model_path = (
            self._make_floating_urdf(self.urdf_path, floating_base_child_link)
            if floating_base
            else self._make_fixed_base_urdf(self.urdf_path)
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.xml_path = self._save_debug_xml(self.model)
        self.data = mujoco.MjData(self.model)
        self.controlled_joints = controlled_joints
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in controlled_joints]
        )
        missing = [j for j, jid in zip(controlled_joints, self.joint_ids) if jid < 0]
        if missing:
            raise ValueError(f"URDF missing controlled joints: {missing}")
        self.qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.dof_ids = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int)
        self.lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.upper = self.model.jnt_range[self.joint_ids, 1].copy()
        limited = self.model.jnt_limited[self.joint_ids] != 0
        self.lower[~limited] = -np.inf
        self.upper[~limited] = np.inf
        self.name_to_index = {name: i for i, name in enumerate(controlled_joints)}
        self.free_qpos_id = -1
        self.free_qvel_id = -1
        if floating_base:
            free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
            if free_id >= 0:
                self.free_qpos_id = int(self.model.jnt_qposadr[free_id])
                self.free_qvel_id = int(self.model.jnt_dofadr[free_id])
        mujoco.mj_forward(self.model, self.data)

    def _make_fixed_base_urdf(self, urdf_path: Path) -> Path:
        """Rewrite mesh paths to absolute ones for a fixed-base URDF.

        Needed because a `<mujoco><compiler meshdir="meshes">` tag would be
        prepended to filenames that are already relative to the URDF, yielding
        "meshes/meshes/...". Resolving to absolute paths and dropping meshdir
        makes loading independent of both the tag and the working directory.
        The floating-base path does the same rewrite as a side effect.
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        changed = False
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.set("filename", str(_resolve_mesh_path(urdf_path, filename)))
                changed = True
        for mujoco_tag in root.findall("mujoco"):
            for compiler in mujoco_tag.findall("compiler"):
                if compiler.attrib.pop("meshdir", None) is not None:
                    changed = True
        if not changed:
            return urdf_path
        out_dir = Path(tempfile.gettempdir()) / "robot_teleop_mujoco"
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = out_dir / f"{urdf_path.stem}_fixed.urdf"
        tree.write(xml_path, encoding="unicode")
        return xml_path

    def _make_floating_urdf(self, urdf_path: Path, child_link: str | None = None) -> Path:
        """Rewrite the URDF with a floating base joint and absolute mesh paths."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.set("filename", str(_resolve_mesh_path(urdf_path, filename)))
        for mujoco_tag in root.findall("mujoco"):
            for compiler in mujoco_tag.findall("compiler"):
                compiler.attrib.pop("meshdir", None)
        for joint in root.findall("joint"):
            if joint.attrib.get("name") == "floating_base_joint":
                return urdf_path

        links = [link.attrib["name"] for link in root.findall("link")]
        child_links: set[str] = set()
        world_child_joint = None
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None:
                continue
            child_name = child.attrib["link"]
            child_links.add(child_name)
            if parent is not None and parent.attrib.get("link") == "world":
                if child_link is None:
                    child_link = child_name
                    world_child_joint = joint
                elif child_name == child_link:
                    world_child_joint = joint

        if child_link is None:
            roots = [name for name in links if name not in child_links and name != "world"]
            if len(roots) != 1:
                raise ValueError(f"Cannot infer floating base child link from URDF roots: {roots}")
            child_link = roots[0]
        if child_link not in links:
            raise ValueError(f"floating_base_child_link not found in URDF: {child_link}")

        if "world" not in links:
            world = ET.Element("link", {"name": "world"})
            root.insert(1, world)
        if world_child_joint is not None:
            root.remove(world_child_joint)

        floating = ET.Element("joint", {"name": "floating_base_joint", "type": "floating"})
        ET.SubElement(floating, "parent", {"link": "world"})
        ET.SubElement(floating, "child", {"link": child_link})
        root.insert(2, floating)

        out_dir = Path(tempfile.gettempdir()) / "robot_teleop_mujoco"
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = out_dir / f"{urdf_path.stem}_floating.urdf"
        tree.write(xml_path, encoding="unicode")
        return xml_path

    def _save_debug_xml(self, model: mujoco.MjModel) -> Path:
        out_dir = Path(tempfile.gettempdir()) / "robot_teleop_mujoco"
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = out_dir / "compiled.xml"
        mujoco.mj_saveLastXML(str(xml_path), model)
        return xml_path

    def integrate_base_velocity(self, vx: float, vy: float, vyaw: float, dt: float) -> None:
        """Dead-reckon the floating base from a velocity command (visualization only)."""
        if self.free_qpos_id < 0:
            return
        yaw = float(
            np.arctan2(
                2.0
                * (
                    self.data.qpos[self.free_qpos_id + 3] * self.data.qpos[self.free_qpos_id + 6]
                    + self.data.qpos[self.free_qpos_id + 4] * self.data.qpos[self.free_qpos_id + 5]
                ),
                1.0
                - 2.0
                * (
                    self.data.qpos[self.free_qpos_id + 5] ** 2
                    + self.data.qpos[self.free_qpos_id + 6] ** 2
                ),
            )
        )
        c = np.cos(yaw)
        s = np.sin(yaw)
        self.data.qpos[self.free_qpos_id + 0] += (c * vx - s * vy) * dt
        self.data.qpos[self.free_qpos_id + 1] += (s * vx + c * vy) * dt
        yaw += vyaw * dt
        half = 0.5 * yaw
        self.data.qpos[self.free_qpos_id + 3 : self.free_qpos_id + 7] = np.array(
            [np.cos(half), 0.0, 0.0, np.sin(half)]
        )
        mujoco.mj_forward(self.model, self.data)

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Unknown body: {body_name}")
        return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def qpos_controlled(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def set_qpos_controlled(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_ids] = clamp_to_limits(
            np.asarray(q, dtype=float), self.lower, self.upper
        )
        mujoco.mj_forward(self.model, self.data)

    def set_joint_qpos(self, joint_names: list[str], q: np.ndarray) -> None:
        values = np.asarray(q, dtype=float)
        if len(joint_names) != len(values):
            raise ValueError("joint_names and q length mismatch")
        for name, value in zip(joint_names, values):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"URDF missing joint: {name}")
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_id] = value
        mujoco.mj_forward(self.model, self.data)

    def joint_qpos(self, joint_names: list[str]) -> np.ndarray:
        values: list[float] = []
        missing: list[str] = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                missing.append(name)
                continue
            values.append(float(self.data.qpos[self.model.jnt_qposadr[joint_id]]))
        if missing:
            raise ValueError(f"URDF missing joints: {missing}")
        return np.asarray(values, dtype=float)

    def arm_command_14(self, left_arm_joints: list[str], right_arm_joints: list[str]) -> np.ndarray:
        q = self.qpos_controlled()
        return np.array([q[self.name_to_index[j]] for j in left_arm_joints + right_arm_joints], dtype=float)
