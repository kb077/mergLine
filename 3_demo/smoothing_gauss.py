import trimesh
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from trimesh import proximity
def smmothing ():
    path = 'output'
    j=99
    mesh_path= path+'\\'+str(j)+'-ring_interpolated.obj'
    partition_path = path+'\\'+str(j)+'-partition.obj'
    mesh_partition = trimesh.load(partition_path)
    mesh1= trimesh.load(mesh_path)
    point_cloud =mesh1.vertices
    sigma = 10.0  
    point_cloud = np.append(point_cloud,mesh1.vertices[:int(sigma*1.7)],axis=0)


    smoothed_point_cloud = np.zeros_like(point_cloud)
    for axis in range(point_cloud.shape[1]):
        smoothed_point_cloud[:, axis] = gaussian_filter(point_cloud[:, axis], sigma=sigma)
    print(path+'\\'+str(j)+'-snake_smoothing2.obj')
    with open(path+'\\'+str(j)+'-snake_smoothing10.obj', 'w') as obj_file:
        for k, xyz in enumerate(smoothed_point_cloud):
            obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')

    nearest_points, distance,triangle_id= proximity.closest_point_naive(mesh_partition,smoothed_point_cloud)

    with open(path+'\\'+str(j)+'-snake_smoothing10_proximity.obj', 'w') as obj_file:
        for k, xyz in enumerate(nearest_points):
            obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
