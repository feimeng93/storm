#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import numpy as np
from ...geom.sos.robot_world import RobotWorldCollisionSos

# import matlab.engine
# eng = matlab.engine.start_matlab()


class SosCollisionCost():# deprecated, we use risk_filter.py instead
    # ideally, we use sos programming in parallel to return the birary results of collision, however, it is not possible to run symbolic math in parallel at a fast speed
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(SosCollisionCost, self).__init__()
        
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
        
    

    

    def risk_cost(self, link_pos_seq, link_rot_seq):
        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        dist = self.robot_world_coll.check_robot_sos_collisions(link_pos_batch,
                                                                   link_rot_batch)
        dist = dist.view(batch_size, horizon, n_links)#, self.n_world_objs)
        # cost only when dist is less
        dist += self.distance_threshold

        dist[dist <= 0.0] = 0.0
        dist[dist > 0.2] = 0.2
        dist = dist / 0.25
        
        cost = torch.sum(dist, dim=-1)


        cost = self.weight * cost 

        return cost.to(inp_device)