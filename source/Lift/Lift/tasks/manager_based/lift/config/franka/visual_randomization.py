# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""Visual randomization configurations.

Camera setup (updated 2026): exactly TWO views are used end-to-end for ACT:
  - ``camera``: a fixed forward-facing scene camera in the world, looking at the
    workspace (the robot, table, and cube).
  - ``wrist`` : a camera mounted on the Franka end-effector (``panda_hand``) that
    rigidly follows the hand.

The previous side/top views (``camera2`` / ``camera3``) have been removed. The
resolution and dtype are defined once below and kept consistent across demo
collection (HDF5 image keys), ACT training, and evaluation.
"""

import torch
import random
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


##############################################################################
# CAMERA CONFIG (single source of truth — two views: scene + wrist)
##############################################################################

# RGB resolution captured by every camera. Identical for both cameras and kept
# consistent end-to-end (demo HDF5 -> ACT training -> evaluation).
CAMERA_HEIGHT = 240
CAMERA_WIDTH = 320

# Camera names == HDF5 image keys == ACT ``camera_names``. Order matters: index 0
# is the "primary" image, stored under the backward-compatible ``images`` key.
CAMERA_NAMES = ["camera", "wrist"]


def make_scene_camera() -> CameraCfg:
    """Fixed forward-facing scene camera looking back at the workspace.

    Placed in front of the table (along +X) and rotated 180 deg about Z so the
    optical axis (+X in the 'world' convention) points back toward the
    robot/table. This is the same framing used by the original front camera,
    only re-resolutioned.
    """
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,  # ~60 deg FOV at the ~1 m rig distance
            focus_distance=400.0,
            horizontal_aperture=20.955,
            # Far clip < env_spacing (5 m) so neighbouring envs are clipped out of
            # the render (no cross-env contamination). Workspace is ~1-1.5 m away.
            clipping_range=(0.05, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Rig-mounted 3rd-person view: to the side (-Y) and above the stand,
            # looking down at the workspace (table at (0.5,0,0), grasp ~(0.45,0,0.15)).
            # Like a camera on a side bar of the robot stand, not floating in the room.
            # Pulled ~0.3 m back along the view axis (vs (0.6,-0.9,0.65)) to frame a bit
            # wider; same look direction so the rotation is unchanged.
            pos=(0.643, -1.16, 0.794),
            rot=(0.6261, -0.1893, 0.1604, 0.7393),  # look-at workspace, up=world+Z
            convention="world",
        ),
    )


def make_wrist_camera() -> CameraCfg:
    """Wrist camera rigidly mounted on the Franka end-effector (``panda_hand``).

    The camera prim is created as a child of ``panda_hand`` so it follows the
    hand. The offset below is RELATIVE TO the ``panda_hand`` frame.

    Pose chosen empirically (see ``scripts/batch_cam.py`` + ``verify_wrist.py``):
    at the grasp, the cube sits ~9.5 cm out along the hand's local +Z (the
    approach axis), at hand-frame ~(0.007, -0.001, 0.095). Mounting the camera
    ~14 cm off the hand along its local -X axis and aiming it back at that grasp
    point frames the cube dead-centre with BOTH gripper fingers visible (a clean
    eye-in-hand view). The previous point-blank (0,0,0.05) pose just saw a wall
    of cube face + occlusion. The rotation below is the look-at orientation
    (optical +X -> view direction, up = world +Z) verified by rendering the real
    parented camera at the lifted pose.
    """
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # Intel RealSense D455 color (OV9782) intrinsics, read from the Isaac
            # asset rsd455.usd -> ~90 deg HFOV, matching real wrist-cam hardware.
            focal_length=1.93,
            focus_distance=400.0,
            horizontal_aperture=3.896,
            # Far clip < env_spacing (5 m) so neighbouring envs are clipped out of
            # the render entirely (no cross-env contamination). Grasp is <0.2 m away.
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Hardware-style PARALLEL mount: camera bolted to the hand's +X back face
            # with its optical axis aligned to the hand's +Z approach axis (looks
            # straight down the grasp), like a real wrist bracket -- NOT a look-at.
            # Geometry (scripts/explore_scene.py): fingers at +Z~0.058, cube at
            # +Z~0.095, fingers open along +-Y. The wide D455 FOV keeps the grasp in
            # frame despite the back-face offset.
            # z moved -0.03 -> +0.03 (+6cm FORWARD along the view axis, "fwd_b"): a
            # filmstrip sweep (results/cam_compare/) showed this enlarges + centres the
            # cube and its finger-contact through the whole grasp, vs the small/low
            # old framing where the contact sat at the bottom edge. Demos must be
            # re-collected with this pose before training (camera <-> demo coupling).
            pos=(0.05, 0.0, 0.03),
            rot=(0.0, 0.7071, 0.0, 0.7071),  # optical +X -> hand +Z; up -> hand +X
            convention="world",
        ),
    )


##############################################################################
# RANDOMIZATION FUNCTIONS
##############################################################################

def set_random_lighting():
    """Set random lighting per environment."""
    import omni.usd
    from pxr import UsdLux, Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()

    dome_light_prim = stage.GetPrimAtPath("/World/light")
    if dome_light_prim.IsValid():
        dome_light = UsdLux.DomeLight(dome_light_prim)
        dome_light.GetIntensityAttr().Set(0.0)

    first_env_params = None

    envs_prim = stage.GetPrimAtPath("/World/envs")
    if not envs_prim.IsValid():
        return None

    for child in envs_prim.GetChildren():
        env_path = str(child.GetPath())
        env_name = env_path.split("/")[-1]

        intensity = random.uniform(20000.0, 300000.0)
        color = (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
        )

        if first_env_params is None:
            first_env_params = {"intensity": intensity, "color": color}

        light_path = f"{env_path}/EnvLight"
        local_light_pos = (0.5, 0.0, 1.5)

        light_prim = stage.GetPrimAtPath(light_path)
        if light_prim.IsValid():
            sphere_light = UsdLux.SphereLight(light_prim)
            sphere_light.GetIntensityAttr().Set(intensity)
            sphere_light.GetColorAttr().Set(Gf.Vec3f(*color))
        else:
            sphere_light = UsdLux.SphereLight.Define(stage, light_path)
            sphere_light.GetRadiusAttr().Set(0.3)
            sphere_light.GetIntensityAttr().Set(intensity)
            sphere_light.GetColorAttr().Set(Gf.Vec3f(*color))
            xformable = UsdGeom.Xformable(sphere_light.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(*local_light_pos))

    return first_env_params

def set_random_texture():
    """Set random cube material properties per environment."""
    import omni.usd
    from pxr import UsdShade, Sdf, Gf

    stage = omni.usd.get_context().get_stage()

    first_env_params = None

    # Find all environment cube objects and override their materials
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        # Match the Object prim in each env: /World/envs/env_*/Object
        if prim_path.endswith("/Object") and "/envs/env_" in prim_path:
            # Generate unique random params for each env
            color = (
                random.uniform(0.1, 1.0),
                random.uniform(0.1, 1.0),
                random.uniform(0.1, 1.0),
            )
            roughness = random.uniform(0.1, 0.9)
            metallic = random.uniform(0.0, 0.8)

            # Store first env params for logging
            if first_env_params is None:
                first_env_params = {"color": color, "roughness": roughness, "metallic": metallic}

            # Extract env name from path (e.g., env_0)
            env_name = prim_path.split("/envs/")[1].split("/")[0]
            mat_path = f"/World/envs/{env_name}/CubeMaterial"

            # Create or update material
            existing_mat = stage.GetPrimAtPath(mat_path)
            if existing_mat.IsValid():
                # Update existing shader
                shader_path = mat_path + "/Shader"
                shader_prim = stage.GetPrimAtPath(shader_path)
                if shader_prim.IsValid():
                    shader = UsdShade.Shader(shader_prim)
                    diffuse_input = shader.GetInput("diffuseColor")
                    if diffuse_input:
                        diffuse_input.Set(Gf.Vec3f(*color))
                    rough_input = shader.GetInput("roughness")
                    if rough_input:
                        rough_input.Set(roughness)
                    metal_input = shader.GetInput("metallic")
                    if metal_input:
                        metal_input.Set(metallic)
            else:
                # Create new material
                material = UsdShade.Material.Define(stage, mat_path)

                # Create shader
                shader_path = mat_path + "/Shader"
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

                # Connect shader to material outputs
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Get the material to bind
            material = UsdShade.Material.Get(stage, mat_path)

            # Bind material to the Object prim and all its descendants
            object_prim = stage.GetPrimAtPath(prim_path)
            if object_prim.IsValid() and material:
                # Apply binding API and bind
                bindable = UsdShade.MaterialBindingAPI.Apply(object_prim)
                bindable.Bind(material, UsdShade.Tokens.strongerThanDescendants)

                # Also bind to geometry/mesh if it exists
                mesh_path = prim_path + "/geometry/mesh"
                mesh_prim = stage.GetPrimAtPath(mesh_path)
                if mesh_prim.IsValid():
                    mesh_bindable = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                    mesh_bindable.Bind(material, UsdShade.Tokens.strongerThanDescendants)

    return first_env_params


def randomize_lighting(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple = (300.0, 12000.0),
    color_temp_range: tuple = (0.0, 1.0),
):
    """Randomize dome light intensity and color temperature at each reset.

    This function modifies the USD light properties directly.
    """
    import omni.usd
    from pxr import UsdLux, Gf

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")

    if light_prim.IsValid():
        dome_light = UsdLux.DomeLight(light_prim)

        # EXTREME intensity range
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        dome_light.GetIntensityAttr().Set(intensity)

        # EXTREME color range: warm orange/red to cool blue
        temp = random.uniform(color_temp_range[0], color_temp_range[1])
        color = (
            1.0 - 0.6 * temp,   # R: 1.0 (warm) -> 0.4 (cool)
            0.6 + 0.2 * temp,   # G: 0.6 -> 0.8
            0.3 + 0.7 * temp,   # B: 0.3 (warm) -> 1.0 (cool)
        )
        dome_light.GetColorAttr().Set(Gf.Vec3f(*color))


def randomize_camera_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    position_noise: tuple = (0.1, 0.1, 0.1),
    rotation_noise: float = 5.0,  # degrees
):
    """Randomize camera position and orientation."""
    pass  # Placeholder - actual implementation via camera offset randomization


def randomize_rigid_body_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    color_range: dict,
):
    """Randomize the color of a rigid body asset.

    Note: This is a simplified version. Full texture randomization
    requires USD material manipulation which is more complex.
    """
    # This function will be called at each reset
    # For full implementation, you'd modify USD materials
    pass


##############################################################################
# CONDITION 1: BASELINE (No visual randomization)
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Baseline(FrankaCubeLiftEnvCfg):
    """Baseline: fixed lighting, textures, camera position (scene + wrist views)."""

    def __post_init__(self):
        super().__post_init__()

        # Reduce envs for camera rendering; widen spacing so the scene camera
        # does not see neighbouring environments.
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0

        # Two views: fixed scene camera + wrist camera on panda_hand.
        self.scene.camera = make_scene_camera()
        self.scene.wrist = make_wrist_camera()

        # Disable debug visualization (target marker)
        self.commands.object_pose.debug_vis = False

        # Fixed lighting (already in base config):
        # light intensity 3000.0, color (0.75, 0.75, 0.75)


##############################################################################
# CONDITION 2: LIGHTING RANDOMIZATION
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Lighting(FrankaCubeLiftEnvCfg):
    """Lighting randomization: intensity, direction, color temperature.

    Lighting is randomized per-demo via set_random_lighting() called from the
    demo generation script, not via reset events.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0

        # Same two views as baseline.
        self.scene.camera = make_scene_camera()
        self.scene.wrist = make_wrist_camera()

        self.commands.object_pose.debug_vis = False

        # Note: lighting is randomized per-demo in generate_demos.py,
        # NOT via reset events (which would change too frequently).


##############################################################################
# CONDITION 3: TEXTURE RANDOMIZATION
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Texture(FrankaCubeLiftEnvCfg):
    """Texture randomization: block materials, table surface (scene + wrist views)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0

        # Same two views as baseline.
        self.scene.camera = make_scene_camera()
        self.scene.wrist = make_wrist_camera()

        self.commands.object_pose.debug_vis = False

        # Add texture randomization event
        self.events.randomize_object_color = EventTerm(
            func=randomize_rigid_body_color,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "color_range": {
                    "r": (0.3, 1.0),
                    "g": (0.3, 1.0),
                    "b": (0.3, 1.0),
                },
            },
        )


##############################################################################
# CONDITION 4: COMBINED (Lighting + Texture randomizations)
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Combined(FrankaCubeLiftEnvCfg):
    """Combined: lighting and texture randomizations together (scene + wrist views)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0

        # Same two views as baseline.
        self.scene.camera = make_scene_camera()
        self.scene.wrist = make_wrist_camera()

        self.commands.object_pose.debug_vis = False

        # Note: lighting and texture are randomized per-demo in generate_demos.py
        # via set_random_lighting() and set_random_texture().
