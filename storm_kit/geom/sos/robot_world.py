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
import copy
import torch

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, rpy_angles_to_matrix, multiply_transform, transform_point
from ...geom.sdf.primitives import sdf_capsule_to_sphere
from .robot import RobotCapsuleCollision, RobotMeshCollision, RobotSphereCollision
from .world import WorldPointCloudCollision, WorldPrimitiveCollision

import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from forward_occupancy.SO import make_spheres
from distance_net.batched_distance_and_gradient_net import BatchedDistanceGradientNet


class RobotWorldCollision:
    def __init__(self, robot_collision, world_collision):
        self.tensor_args = robot_collision.tensor_args
        self.robot_coll = robot_collision
        self.world_coll = world_collision
    def update_robot_link_poses(self, links_pos, links_rot):
        self.robot_coll.update_robot_link_poses(links_pos, links_rot)
    def update_world_robot_pose(self, w_pos, w_rot):
        self.world_coll.update_reference_frame(w_pos, w_rot)
    
class RobotWorldCollisionSos(RobotWorldCollision):  
    def __init__(self, robot_collision_params, world_collision_params, robot_batch_size=1, world_batch_size=1,tensor_args={'device':"cpu", 'dtype':torch.float32}, bounds=None, grid_resolution=None, max_spheres=5000, n_params=7, self_collision_top_k=5):
        self.dimension = robot_collision_params['dimension']
        self.n_params = robot_collision_params['n_params']
        self.dtype = tensor_args['dtype']
        self.np_dtype = torch.empty(0,dtype=self.dtype).numpy().dtype
        if device is None:
            device = torch.empty(0,dtype=self.dtype).device
        self.device = tensor_args['device']
        self.top_k = self_collision_top_k # hyperparameter, 5-10 seems to be a good number
        self.distance_net = BatchedDistanceGradientNet().to(self.device).eval()
        self.__allocate_base_params(max_spheres)

    def __allocate_base_params(self, max_spheres):
        self.max_spheres = max_spheres
        self.__centers = torch.empty((self.max_spheres, self.dimension), dtype=self.dtype, device=self.device)
        self.__radii = torch.empty((self.max_spheres), dtype=self.dtype, device=self.device)
        self.__center_jac = torch.empty((self.max_spheres, self.dimension, self.n_params), dtype=self.dtype, device=self.device)
        self.__radii_jac = torch.empty((self.max_spheres, self.n_params), dtype=self.dtype, device=self.device)

    def set_params(self, centers_bpz, radii, p_idx, obs_link_mask, g_ka, n_spheres_per_link, n_time, obs_tuple, self_collision):
        self.centers_bpz = centers_bpz
        self.p_idx = p_idx
        self.obs_link_mask = obs_link_mask
        self.g_ka = g_ka

        ### Self collision checks
        if self_collision is not None:
            self.self_collision_link_link = self_collision[0]
            self.self_collision_link_joint = self_collision[1]
            self.self_collision_joint_joint = self_collision[2]
            self.n_link_link = self.self_collision_link_link.count_nonzero(dim=1).cpu()
            self.n_link_joint = self.self_collision_link_joint.count_nonzero(dim=1).cpu()
            self.n_joint_joint = self.self_collision_joint_joint.count_nonzero(dim=1).cpu()
            num_link_link = self.n_link_link.sum()
            num_link_joint = self.n_link_joint.sum()
            num_joint_joint = self.n_joint_joint.sum()
            # self.n_self_collision = num_link_link*n_time*n_spheres_per_link*n_spheres_per_link \
            #                         + num_link_joint*n_time*n_spheres_per_link \
            #                         + num_joint_joint*n_time
            self.n_self_collision_full = num_link_link*n_spheres_per_link*n_spheres_per_link \
                                    + num_link_joint*n_spheres_per_link \
                                    + num_joint_joint
            self.n_self_collision = self.top_k*n_time
        else:
            self.n_self_collision = 0
        ###

        n_joints = centers_bpz.batch_shape[0]
        n_pairs = p_idx.shape[-1]
        if obs_link_mask is not None:
            if self.n_self_collision == 0:
                self.p_idx = p_idx[:,obs_link_mask]
            n_obs_link_pairs = torch.count_nonzero(obs_link_mask).item()
        else:
            n_obs_link_pairs = n_pairs
        
        self.n_spheres = n_spheres_per_link
        self.n_time = n_time
        self.n_joints = n_joints
        self.total_spheres = n_joints*n_time + n_spheres_per_link*n_obs_link_pairs*n_time
        if self.total_spheres > self.max_spheres:
            # reallocate new tensors
            self.__allocate_base_params(self.total_spheres)

        ### num constraints
        self.M = self.total_spheres + self.n_self_collision
        ###

        ## Utilize underlying storage and update initial radii values
        self.centers = self.__centers[:self.total_spheres]
        self.radii = self.__radii[:self.total_spheres]
        self.center_jac = self.__center_jac[:self.total_spheres]
        self.radii_jac = self.__radii_jac[:self.total_spheres]

        self.radii.view(-1, self.n_time)[:n_joints] = radii
        self.radii_jac[:self.n_time*n_joints] = 0

        ## Obstacle data
        self.obs_tuple = obs_tuple


    def __self_collision(self, spheres, joint_centers, joint_radii, joint_jacs, Cons_out, Jac_out):
        '''This function is used to compute the distance and gradient of the distance to each other sphere that is valid for self-collision.

        Args:
            spheres: A tuple of tensors representing the centers and radii of each sphere and their respective jacobians.
            joint_centers: A tensor of shape (n_joints, n_time, dim) representing the centers of each joint.
            joint_radii: A tensor of shape (n_joints, n_time) representing the radii of each joint.
            joint_jacs: A tensor of shape (n_joints, n_time, dim, n_params) representing the jacobians of each joint.
            Cons_out: An output tensor of shape (n_self_collision) representing the constraint values. This is modified in place.
            Jac_out: An output tensor of shape (n_self_collision, n_params) representing the jacobian of the constraint values. This is modified in place.
        '''
        # First get each link to link distance
        # spheres[0] are (link_idx, time_idx, sphere_idx, dim)
        # spheres[1] are (link_idx, time_idx, sphere_idx)
        # link_spheres_c are (time_idx, sphere_idx, dim)
        # link_spheres_r are (time_idx, sphere_idx)
        Cons_out_ = torch.empty((self.n_time, self.n_self_collision_full), dtype=self.dtype, device=self.device)
        Jac_out_ = torch.empty((self.n_time, self.n_self_collision_full, self.n_params), dtype=self.dtype, device=self.device)
        sidx = 0
        for link_idx, (link_spheres_c, link_spheres_r, jac_c, jac_r) in enumerate(zip(*spheres)):
            if self.n_link_link[link_idx] == 0:
                continue
            comp_spheres_c = spheres[0][self.self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx, dim)
            comp_spheres_r = spheres[1][self.self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx)
            delta = link_spheres_c[None,...,None,:] - comp_spheres_c[...,None,:,:]
            dists = torch.linalg.vector_norm(delta, dim=-1)
            surf_dists = dists - link_spheres_r.unsqueeze(-1) - comp_spheres_r.unsqueeze(-2) # (n_comp_links, time_idx, sphere_idx(self), sphere_idx)
            # compute jacobian
            # delta is (n_comp_links, time_idx, sphere_idx(self), sphere_idx, dim)
            # jac_c is (n_comp_links, time_idx, sphere_idx(self), dim, n_params)
            # jac_r is (n_comp_links, time_idx, sphere_idx(self), n_params)
            comp_spheres_jac_c = spheres[2][self.self_collision_link_link[link_idx]]
            comp_spheres_jac_r = spheres[3][self.self_collision_link_link[link_idx]]
            # (n_comp_links, time_idx, sphere_idx(self), sphere_idx, dim, n_params)
            jac_dists_inner = (delta/dists.unsqueeze(-1)).unsqueeze(-1) * (jac_c[None,...,None,:,:] - comp_spheres_jac_c[...,None,:,:,:])
            # (n_comp_links, time_idx, sphere_idx(self), sphere_idx, n_params)
            jac_dists = torch.sum(jac_dists_inner, dim=-2)
            jac_surf_dists = jac_dists - jac_r.unsqueeze(-2) - comp_spheres_jac_r.unsqueeze(-3)
            # Save
            eidx = sidx + surf_dists.numel()//self.n_time
            Cons_out_[:,sidx:eidx].copy_(-surf_dists.transpose(0,1).reshape(self.n_time, -1))
            Jac_out_[:,sidx:eidx].copy_(-jac_surf_dists.transpose(0,1).reshape(self.n_time, -1, self.n_params))
            sidx = eidx


    def build_constraints(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if Cons_out is None:
            Cons_out = np.empty(self.M, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M, self.n_params), dtype=self.np_dtype)
        Cons_out = torch.from_numpy(Cons_out)
        Jac_out = torch.from_numpy(Jac_out)

        # Batch form of batch construction
        centers = self.centers.view(-1, self.n_time, self.dimension)
        center_jac = self.center_jac.view(-1, self.n_time, self.dimension, self.n_params)
        radii = self.radii.view(-1, self.n_time)
        centers[:self.n_joints] = self.centers_bpz.center_slice_all_dep(x)
        center_jac[:self.n_joints] = self.centers_bpz.grad_center_slice_all_dep(x)
        joint_centers = centers[self.p_idx]
        joint_radii = radii[self.p_idx]
        joint_jacs = center_jac[self.p_idx]
        spheres = make_spheres(joint_centers[0], joint_centers[1], joint_radii[0], joint_radii[1], joint_jacs[0], joint_jacs[1], self.n_spheres)
        sidx = self.n_joints * self.n_time
        if self.n_self_collision > 0 and self.obs_link_mask is not None:
            self.centers[sidx:] = spheres[0][self.obs_link_mask].reshape(-1,self.dimension)
            self.radii[sidx:] = spheres[1][self.obs_link_mask].reshape(-1)
            self.center_jac[sidx:] = spheres[2][self.obs_link_mask].reshape(-1,self.dimension,self.n_params)
            self.radii_jac[sidx:] = spheres[3][self.obs_link_mask].reshape(-1,self.n_params)
        else:
            self.centers[sidx:] = spheres[0].reshape(-1,self.dimension)
            self.radii[sidx:] = spheres[1].reshape(-1)
            self.center_jac[sidx:] = spheres[2].reshape(-1,self.dimension,self.n_params)
            self.radii_jac[sidx:] = spheres[3].reshape(-1,self.n_params)

        # Do what you need with the centers and radii
        # NN(centers) - r > 0
        # D_NN(c(k)) -> D_c(NN) * D_k(c) - D_k(r)
        # D_c(NN) should have shape (n_spheres, 3, 1)
        # D_k(c) has shape (n_spheres, 3, n_params)
        # D_k(r) has shape (n_spheres, n_params)
        dist, dist_jac = self.NN_fun(self.centers, self.obs_tuple)
        cons_dists_out = Cons_out[:self.total_spheres]
        cons_dists_jac_out = Jac_out[:self.total_spheres]
        cons_dists_out.copy_(-(dist - self.radii))
        cons_dists_jac_out.copy_(-((dist_jac.unsqueeze(-1) * self.center_jac).sum(1) - self.radii_jac))

        ### Add self collision if enabled
        if self.n_self_collision > 0:
            self_collision_dists = Cons_out[self.total_spheres:]
            self_collision_jacs = Jac_out[self.total_spheres:]
            joint_centers_all = centers[:self.n_joints]
            joint_radii_all = radii[:self.n_joints]
            joint_jacs_all = center_jac[:self.n_joints]
            self.__self_collision(spheres, joint_centers_all, joint_radii_all, joint_jacs_all, self_collision_dists, self_collision_jacs)
        
        return Cons_out, Jac_out
    

    def check_robot_sos_collisions(self, link_pos):
        if len(link_pos.shape) == 1:    
            spheres = self.build_constraints(link_pos)

            ellipse = self.build_ellipsoids(spheres)

            # for matlab
            # ellipse_centers = np.zeros((time_steps,3,3))
            # for i , j in enumerate([1,3,5]):
            #     ellipse_centers[:,i,:]=(link_centers[:,j,:]+link_centers[:,j+1,:])/2
            # check_interval = self.t_to_peak.cpu().numpy()[1:]
            # status = eng.risk_verification(matlab.double(ellipse_centers.tolist()),matlab.double(check_interval.tolist()))
        elif len(link_pos.shape) == 2:
            pass
        else:
            raise ValueError("link_pos must be 1D [dof] or 2D [horizon, dof] tensor for collision risk check")

        return status
    

    def collision_risk_cost(self):
        time_steps = len(R_q[0])
        link_centers = np.zeros((time_steps,n_links,3))

        R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
        for j in range(self.n_links):
            P = R@self.P0[j] + P
            R = R@self.R0[j]@R_q[:,j]
            link = batchZonotope(self.link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1))
            link = R@link+P
            links.append(link)
            link_centers[:,j,:]=link.center.cpu().detach().numpy()

        SFO_link = self.SFO.construct_sphere()
        status = self.robot_world_coll.risk_check(SFO_link)


    def risk_veri_interval(self,link_pos):
        # link_pos : [(7,)], 7: joint angles
        inp_device = link_pos.device
        batch_size = link_pos.shape[0]
        time_steps = link_pos.shape[1]
        n_links = link_pos.shape[2]
 
        links = []

        R_q = self.rot(link_pos)

        time_steps = 1
        R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
        for j in range(self.n_links):
            P = R@self.P0[j] + P
            R = R@self.R0[j]@R_q[j]
            link = R@self.__link_zonos[j]+P
            links.append(link)
            link_centers[:,j,:]=link.center.cpu().detach().numpy()

        SFO_link = self.SFO.construct_sphere()
        status = self.robot_world_coll.risk_check(SFO_link)
        
        ellipse_centers = np.zeros((time_steps,3,3))
        for i , j in enumerate([1,3,5]):
            ellipse_centers[:,i,:]=(link_centers[:,j,:]+link_centers[:,j+1,:])/2
        check_interval = self.t_to_peak.cpu().numpy()[1:]
        
        status = eng.risk_verification(matlab.double(ellipse_centers.tolist()),matlab.double(check_interval.tolist()))
        #size(batch, time)
        cost = self.weight * (status-1) * (-0.8)

        return cost.to(inp_device)
        
        

class RobotWorldCollisionPrimitive(RobotWorldCollision):
    def __init__(self, robot_collision_params, world_collision_params, robot_batch_size=1,
                 world_batch_size=1,tensor_args={'device':"cpu", 'dtype':torch.float32},
                 bounds=None, grid_resolution=None):
        robot_collision = RobotSphereCollision(robot_collision_params, robot_batch_size, tensor_args)

        
        world_collision = WorldPrimitiveCollision(world_collision_params, tensor_args=tensor_args, batch_size=world_batch_size, bounds=bounds, grid_resolution=grid_resolution)
        self.robot_batch_size = robot_batch_size

        super().__init__(robot_collision, world_collision)
        self.dist = None

    def build_batch_features(self, batch_size, clone_pose=True, clone_points=True):
        self.batch_size = batch_size
        self.robot_coll.build_batch_features(clone_objs=clone_points, batch_size=batch_size)

    def check_robot_sphere_collisions(self, link_trans, link_rot):
        """get signed distance from stored grid [very fast]

        Args:
            link_trans (tensor): [b,3]
            link_rot (tensor): [b,3,3]

        Returns:
            tensor: signed distance [b,1]
        """        
        batch_size = link_trans.shape[0]
        # update link pose:
        if(self.robot_batch_size != batch_size):
            self.robot_batch_size = batch_size
            self.build_batch_features(self.robot_batch_size, clone_pose=True, clone_points=True)

        self.robot_coll.update_batch_robot_collision_objs(link_trans, link_rot)

        w_link_spheres = self.robot_coll.get_batch_robot_link_spheres()
        
                
        
        n_links = len(w_link_spheres)

        if(self.dist is None or self.dist.shape[0] != n_links):
            self.dist = torch.zeros((batch_size, n_links), **self.tensor_args)
        dist = self.dist

        for i in range(n_links):
            spheres = w_link_spheres[i]
            b, n, _ = spheres.shape
            spheres = spheres.view(b * n, 4)

            # compute distance between world objs and link spheres
            sdf = self.world_coll.check_pts_sdf(spheres[:,:3]) + spheres[:,3]
            sdf = sdf.view(b,n)
            dist[:,i] = torch.max(sdf, dim=-1)[0]
 
        return dist



        
    def get_robot_env_sdf(self, link_trans, link_rot):
        """Compute signed distance via analytic functino

        Args:
            link_trans (tensor): [b,3]
            link_rot (tensor): [b,3,3]

        Returns:
            tensor : signed distance [b,1]
        """        
        batch_size = link_trans.shape[0]
        # update link pose:
        if(self.robot_batch_size != batch_size):
            self.robot_batch_size = batch_size
            self.build_batch_features(self.robot_batch_size, clone_pose=True, clone_points=True)

        self.robot_coll.update_batch_robot_collision_objs(link_trans, link_rot)

        w_link_spheres = self.robot_coll.get_batch_robot_link_spheres()
        
                
        
        n_links = len(w_link_spheres)

        if(self.dist is None or self.dist.shape[0] != n_links):
            self.dist = torch.empty((batch_size, n_links), **self.tensor_args)
        dist = self.dist

        

        
        for i in range(n_links):
            spheres = w_link_spheres[i]
            #b, n, _ = spheres.shape
            #spheres = spheres.view(b * n, 4)

            # compute distance between world objs and link spheres
            d = self.world_coll.get_sphere_distance(spheres)
            
            dist[:,i] = torch.max(torch.max(d, dim=-1)[0], dim=-1)[0]

        return dist

    
