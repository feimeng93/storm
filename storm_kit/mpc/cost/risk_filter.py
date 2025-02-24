import torch
import numpy as np
from ...geom.sos.robot_world import RobotWorldCollisionSos


class RiskFilter():
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(RiskFilter, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        robot_collision_params = robot_params['robot_collision_params']
        self.batch_size = -1
        # BUILD world and robot:
        self.robot_world_coll = RobotWorldCollisionSos(robot_collision_params,
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'])
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        
    

    def rot(self,q=None):
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3,dtype=self.dtype,device=self.device) + torch.sin(q)*self.rot_skew_sym + (1-torch.cos(q))*self.rot_skew_sym@self.rot_skew_sym
    
    
    def risk_veri(self, link_pos):
        inp_device = link_pos.device
        status = self.robot_world_coll.check_robot_sos_collisions_step(link_pos)
        return status
    
    def risk_cost_regul(self):
        dist = dist.view(batch_size, horizon, n_links)#, self.n_world_objs)
        # cost only when dist is less
        dist += self.distance_threshold

        dist[dist <= 0.0] = 0.0
        dist[dist > 0.2] = 0.2
        dist = dist / 0.25
        
        cost = torch.sum(dist, dim=-1)
        if cost != 0:
            cost = self.risk_cost_regul()
        cost = self.weight * cost 
        

