clc
clear all
close all
addpath('plotcube')

currentFile = mfilename('fullpath');
parent_path = fileparts(currentFile);
rootFile = fileparts(parent_path);

save_path = fullfile(parent_path,'franka');
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

% % robot_path = fullfile(fileparts(rootFile),'assets','robots','kinova_arm','gen3.urdf');
robot_path = fullfile(rootFile,'assets','robots','franka_description','franka_panda_no_gripper.urdf');
robot = importrobot(robot_path);

n_bodies = robot.NumBodies;
link = robot.Base;
children = link.Children;        
link = children{1,1};
R = eye(3);
xyz = [0 0 0];
RR = cell(n_bodies+1,1); PP = cell(n_bodies+1,1);
for i = 1:n_bodies+1
    mesh_path = '/Users/mengfei/Desktop/RDF/environments/robots/assets/robots/franka_description/meshes/collision/' + strrep(string(link.Collisions), 'Mesh Filename ','');
    %mesh_path = fullfile(rootFile,'assets','robots','fetch_arm','fetch_description','meshes',strrep(string(link.Visuals),'Mesh:',''));
    %mesh_name = strrep(strrep(strrep(string(link.Visuals),'Mesh:',''),'stl',''),'_',' ');
    children = link.Children;
    if size(children,2) >1 || size(children,2) == 0
        break;
    end
        
    link = children{1,1};
    R0 = link.Joint.JointToParentTransform(1:3,1:3); xyz0 = link.Joint.JointToParentTransform(1:3,4)';
    TR = stlread(mesh_path);
    
    Points = (TR.Points)*R+xyz;
    xyz = xyz+xyz0*R;
    R = R0'*R; %[0 -1 0; 1 0 0; 0 0 1]'*
    RR{i} = R'; PP{i} = xyz; 
    xyz_M = max(Points); xyz_m = min(Points);
    c = (xyz_M+xyz_m)/2; g = diag((xyz_M-xyz_m)/2);
    
    figure(1)
    trimesh(TR.ConnectivityList,Points(:,1),Points(:,2),Points(:,3));
    plotcube(2*diag(g)',c-diag(g)',0.2,[1,0,0]);
    hold on
    title('franka with bounding boxes')

    Points =TR.Points;
    xyz_M = max(TR.Points); xyz_m = min(TR.Points);
    Z = [(xyz_M+xyz_m)'/2,diag((xyz_M-xyz_m)/2)]';
    filename = sprintf('franka_link_zonotope_%i.mat', i-1);
    save(fullfile(save_path,filename),'Z');
    
end
axis("equal")
figure(i+1)
show(robot)