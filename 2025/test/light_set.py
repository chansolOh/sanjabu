import numpy as np
from pxr import  Gf
import omni.usd
import cs_rep_utils as csr
class light_inst:
    def __init__(self,prim):
        self.prim = prim
        self.translate = self.prim.GetAttribute("xformOp:translate").Get()
        self.rotate = self.prim.GetAttribute("xformOp:rotateXYZ").Get()
        self.color = self.prim.GetAttribute("inputs:color").Get()
        self.enable_color_temp = self.prim.GetAttribute("inputs:enableColorTemperature").Get()
        self.color_temp = self.prim.GetAttribute("inputs:colorTemperature").Get()
        self.intensity = self.prim.GetAttribute("inputs:intensity").Get()
        self.exposure = self.prim.GetAttribute("inputs:exposure").Get()
        self.state_dict = {
            "name" : self.prim.GetName(),
            "translate" : np.array(self.translate).tolist(),
            "color" : np.array(self.color).tolist(),
            "enable_color_temp" : self.enable_color_temp,
            "color_temp":self.color_temp,
            "intensity":self.intensity,
            "exposure": self.exposure,
        }



class Light:
    def __init__(self, light_prim_list):
        self.lights = light_prim_list
        self.lights_init_state = [ light_inst(i) for i in self.lights]
        self.lights_scale =[ csr.find_parents_scale(light) for light in self.lights]  #### scale 각각 구해서 값에 변경
    
    def get_all_state(self):
        lights = [ light_inst(i) for i in self.lights]
        return [i.state_dict for i in lights]
        
    def set_each_setting(self,setting_list):
        for set_ls in setting_list:
            for light in self.lights:
                if light.GetName() == set_ls["name"]:
                    light.GetAttribute("xformOp:translate").Set(Gf.Vec3d(set_ls["translate"]))
                    light.GetAttribute("inputs:color").Set(Gf.Vec3f(set_ls["color"]))
                    light.GetAttribute("inputs:enableColorTemperature").Set(set_ls["enable_color_temp"])
                    light.GetAttribute("inputs:colorTemperature").Set(set_ls["color_temp"])
                    light.GetAttribute("inputs:intensity").Set(set_ls["intensity"])
                    light.GetAttribute("inputs:exposure").Set(set_ls["exposure"])
    
    def set_all_exposure(self):
        pass
    def set_all_intensity(self):
        pass
    def set_all_pose(self):
        pass
    
    def add_exposure(self):
        pass
    def add_intensity(self):
        pass
    def add_pose(self):
        pass

    def random_temp(self, val = 500, default_temp = 6000):
        for light,init_val in zip (self.lights, self.lights_init_state):
            random_val = (np.random.random(1)*2-1) *val
            light.GetAttribute("inputs:colorTemperature").Set( (init_val.color_temp if default_temp==None else default_temp) + random_val[0] )
    
    def random_exposure(self, val = 1.0, default_exposure = None):
        for light,init_val in zip (self.lights, self.lights_init_state):
            random_val = (np.random.random(1)*2-1) *val
            light.GetAttribute("inputs:exposure").Set( (init_val.exposure if default_exposure==None else default_exposure) + random_val[0] )
            
    def random_intensity(self, val = 1.0):
        for light,init_val in zip (self.lights, self.lights_init_state):
            random_val = (np.random.random(1)*2-1) *val
            light.GetAttribute("inputs:intensity").Set( init_val.intensity + random_val[0] )
            
    def random_trans(self, val = 0.1 , fixed_axis = []):
        for light,init_val, scale in zip (self.lights, self.lights_init_state, self.lights_scale):
            random_val = (np.random.random(3)*2-1) *val 
            random_val *= (1/scale)
            for ax in fixed_axis:
                random_val[ax] = 0
            try:
                light.GetAttribute("xformOp:translate").Set( init_val.translate + Gf.Vec3d([i for i in random_val]))
            except:
                import pdb; pdb.set_trace()
        