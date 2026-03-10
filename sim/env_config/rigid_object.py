import os
import numpy as np
import torch

import omni.kit.commands
import omni.physxdemos as demo
import isaacsim.core.utils.prims as prims_utils
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from isaacsim.core.api import World
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.nucleus import get_assets_root_path

from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.utils.semantics import add_update_semantics, get_semantics
from isaacsim.core.utils.rotations import euler_angles_to_quat

class RigidObject:
    def __init__(
        self, 
        world:World,
        usd_path: str,
        pos:np.ndarray=np.array([0.0, 0.0, 0.5]), 
        ori:np.ndarray=np.array([0.0, 0.0, 0.0]),
        scale:np.ndarray=np.array([0.0085, 0.0085, 0.0085]),
        visual_material_usd:str |None=None,
        color_material_rgb:tuple|list=None,
        mass: float=0.1,
        static_friction: float=0.8,
        dynamic_friction: float=0.8,
        restitution: float=0.0,
        contact_offset: float=0.01,
        rest_offset: float=0.0,
    ):
        self.world = world
        self.stage = world.stage
        self.scene = world.get_physics_context()._physics_scene

        self.usd_path=usd_path
        self.position = pos
        self.orientation = ori
        self.scale = scale
        self.visual_material_usd = visual_material_usd
        self.color_material_rgb = color_material_rgb


               
        self.rigid_view = UsdGeom.Xform.Define(self.stage, "/World/Rigid")
        self.rigid_name = find_unique_string_name(
            initial_name="rigid",
            is_unique_fn=lambda x: not world.scene.object_exists(x)
        )
        self.rigid_prim_path=find_unique_string_name("/World/Rigid/rigid",is_unique_fn=lambda x: not is_prim_path_valid(x))
        
        # define deformable object Xform
        self.rigid_xform = SingleXFormPrim(
            prim_path=self.rigid_prim_path,
            name=self.rigid_name, 
            position=self.position,
            orientation=euler_angles_to_quat(self.orientation,degrees=True),
            scale=self.scale,
        )

        # add deformable object usd to stage
        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.rigid_prim_path)
        
        self.rigid=SingleRigidPrim(
            prim_path=self.rigid_prim_path,
            name=self.rigid_name,
        )
        self.set_mass(mass)
        
        self.rigid_material_path=find_unique_string_name(self.rigid_prim_path+"/physcis_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.rigid_material=PhysicsMaterial(
            prim_path=self.rigid_material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        
        # set visual material to deformable object
        if self.visual_material_usd is not None:
            self.apply_visual_material(self.visual_material_usd)

        if self.color_material_rgb is not None:
            self.apply_color_visual_material(self.color_material_rgb)

        self.set_contact_offset(contact_offset)  # glove 0.015
        self.set_rest_offset(rest_offset)      # glove 0.01
        
    def  apply_color_visual_material(self,color_rgb):
        """Creates a simple PreviewSurface material with the given color and binds it."""
        material_path = find_unique_string_name(
            initial_name=self.rigid_prim_path + "/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        PreviewSurface(prim_path=material_path, color=np.array(color_rgb))
        
        
        
        
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.rigid_prim_path,
            material_path=material_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )
        # Also bind to any mesh children to ensure override
        root_prim=prims_utils.get_prim_at_path(self.rigid_prim_path)
        children_prims = prims_utils.get_prim_children(root_prim)
        for prim in children_prims:
            if prim.IsA(UsdGeom.Gprim):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=prim.GetPath(),
                    material_path=material_path,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )
    
    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.deformable_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)
        
        root_prim=prims_utils.get_prim_at_path(self.rigid_prim_path)
        children = prims_utils.get_prim_children(root_prim)
        if len(self.children)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.rigid_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.rigid_prim_path, material_path=self.material_prim_path)
            for prim in children:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)
                
    def set_contact_offset(self,contact_offset:float=0.01):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.rigid.prim)
        self.collsionapi.GetContactOffsetAttr().Set(contact_offset)

    def set_rest_offset(self,rest_offset:float=0.000):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.rigid.prim)
        self.collsionapi.GetRestOffsetAttr().Set(rest_offset)
    
    def set_mass(self,mass=0.02):
        physicsUtils.add_mass(self.world.stage, self.rigid_prim_path, mass)
  
    def set_obj_pose(self, pos, ori=None):
        if ori is not None:
            ori = euler_angles_to_quat(ori, degrees=True)
        self.rigid.set_world_pose(pos, ori)

    def get_obj_pos(self):
        return self.rigid.get_world_pose()

      
    