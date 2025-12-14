# Oliver Fritsche
# December 10, 2025
# CS 7180 Advanced Perception

"""Visual randomization configurations."""

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


##############################################################################
# CONDITION 1: BASELINE (No visual randomization)
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Baseline(FrankaCubeLiftEnvCfg):
    """Baseline: Fixed lighting, textures, camera position."""

    def __post_init__(self):
        super().__post_init__()
        
        # Reduce envs for camera rendering
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0  # Increase spacing so cameras don't see other envs
        
        # Camera positioned behind/above robot, looking forward at table
        # Based on Isaac Lab docs example pattern
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,  # Zoomed in to fill frame with robot/table
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),  # Limit far clip to avoid seeing distant envs
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(3.0, 0.0, .2),  # Aligned with robot X-axis, in front of table
                rot=(0.0, 0.0, 0.0, 1.0),  # No rotation, no tilt
                convention="world",
            ),
        )

        # Camera 2: Side view - positioned to the left of the table, looking sideways at it
        self.scene.camera2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera2",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, -4.0, 0.2),  # Beside the table, 1m to the left
                rot=(0.707, 0, 0.0, 0.707),  # 90° right to look at table
                convention="world",
            ),
        )

        self.scene.camera3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera3",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                # clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0, 5),  
                rot=(0.0, -0.707, 0.0, 0.707), 
                convention="world",
            ),
        )
        
        # Disable debug visualization (target marker)
        self.commands.object_pose.debug_vis = False
        
        # Fixed lighting (already in base config)
        # light intensity: 3000.0, color: (0.75, 0.75, 0.75)


##############################################################################
# CONDITION 2: LIGHTING RANDOMIZATION
##############################################################################

@configclass 
class FrankaCubeLiftEnvCfg_Demo_Lighting(FrankaCubeLiftEnvCfg):
    """Lighting randomization: intensity, direction, color temperature.
    
    Lighting is randomized per-demo via set_random_lighting() called from
    the demo generation script, not via reset events.
    """

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        
        # Same camera as baseline
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(3.0, 0.0, 0.2),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="world",
            ),
        )
        
        # Camera 2: Side view - positioned to the left of the table, looking sideways at it
        self.scene.camera2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera2",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, -4.0, 0.2),  # Beside the table, 1m to the left
                rot=(0.707, 0, 0.0, 0.707),  # 90° right to look at table
                convention="world",
            ),
        )

        self.scene.camera3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera3",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                # clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0, 5),  
                rot=(0.0, -0.707, 0.0, 0.707), 
                convention="world",
            ),
        )
        
        # Disable debug visualization
        self.commands.object_pose.debug_vis = False
        
        # Note: Lighting is randomized per-demo in generate_demos.py
        # NOT via reset events (which would change too frequently)


##############################################################################
# CONDITION 3: TEXTURE RANDOMIZATION  
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Texture(FrankaCubeLiftEnvCfg):
    """Texture randomization: block materials, table surface."""

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        
        # Same camera as baseline
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(3.0, 0.0, 0.2),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="world",
            ),
        )
        
        # Disable debug visualization
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

        # Camera 2: Side view - positioned to the left of the table, looking sideways at it
        self.scene.camera2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera2",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, -4.0, 0.2),  # Beside the table, 1m to the left
                rot=(0.707, 0, 0.0, 0.707),  # 90° right to look at table
                convention="world",
            ),
        )

        self.scene.camera3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera3",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                # clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0, 5),  
                rot=(0.0, -0.707, 0.0, 0.707), 
                convention="world",
            ),
        )


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
# CONDITION 4: COMBINED (Lighting + Texture randomizations)
##############################################################################

@configclass
class FrankaCubeLiftEnvCfg_Demo_Combined(FrankaCubeLiftEnvCfg):
    """Combined: Lighting and texture randomizations together (no camera changes)."""

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        
        # Same fixed camera as baseline (no camera randomization)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(3.0, 0.0, 0.2),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="world",
            ),
        )

        # Camera 2: Side view - positioned to the left of the table, looking sideways at it
        self.scene.camera2 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera2",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, -4.0, 0.2),  # Beside the table, 1m to the left
                rot=(0.707, 0, 0.0, 0.707),  # 90° right to look at table
                convention="world",
            ),
        )

        self.scene.camera3 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera3",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=60.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                # clipping_range=(0.1, 6.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0, 5),  
                rot=(0.0, -0.707, 0.0, 0.707), 
                convention="world",
            ),
        )
        
        # Disable debug visualization
        self.commands.object_pose.debug_vis = False
        
        # Note: Lighting and texture are randomized per-demo in generate_demos.py
        # via set_random_lighting() and set_random_texture() functions
