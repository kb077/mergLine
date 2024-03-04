
import numpy as np
def interpolated ():
    path = 'output'
    j=99
    model_name ='pointmlp-full-dataset-biou-alpha'
    print(path+'\\'+str(j)+'-'+model_name+'-prediction-cut+.obj')

    import trimesh 
    open3d_mesh2 = trimesh.load(path+'\\'+str(j)+'-'+model_name+'-prediction-cut.obj')
    
    boundary_list = boundary(open3d_mesh2)
    
    max_list_len = 0
    max_list_idx = 0
    for k, i in enumerate(boundary_list):
        if len(i) >=max_list_len:
            max_list_len = len(i)
            max_list_idx = k

    
    ring  = boundary_list[max_list_idx]
    
    #
    num_interpolations = 2
    interpolated_points = []   
    for  i in range (len(ring)-1):
        start_point = ring[i]
        end_point = ring[i + 1]
        
        for k in range(num_interpolations):
            t = k / (num_interpolations + 1)  
            interpolated_point = (1 - t) * start_point + t * end_point
            interpolated_points.append(interpolated_point)
                    
    start_point = ring[-1]
    end_point = ring[0]
    for k in range(num_interpolations):
        t = k / (num_interpolations + 1)  
        interpolated_point = (1 - t) * start_point + t * end_point
        interpolated_points.append(interpolated_point)
    ring_interpolated = np.array(interpolated_points)    
    with open(path+'\\'+str(j)+'-ring_interpolated.obj', 'w') as obj_file:
        for k, xyz in enumerate(ring_interpolated):
            obj_file.write(f'v {ring_interpolated[k][0]} {ring_interpolated[k][1]} {ring_interpolated[k][2]}\n')    

def boundary(mesh, close_paths=True):
    edge_set = set()
    boundary_edges = set()

    for e in map(tuple, mesh.edges_sorted):
        if e not in edge_set:
            edge_set.add(e)
            boundary_edges.add(e)
        elif e in boundary_edges:
            boundary_edges.remove(e)
        else:
            raise RuntimeError(f"The mesh is not a manifold: edge {e} appears more than twice.")

    from collections import defaultdict
    neighbours = defaultdict(lambda: [])
    for v1, v2 in boundary_edges:
        neighbours[v1].append(v2)
        neighbours[v2].append(v1)

    boundary_paths = []

    while len(boundary_edges) > 0:
        v_previous, v_current = next(iter(boundary_edges))
        boundary_vertices = [v_previous]

        while v_current != boundary_vertices[0]:
            boundary_vertices.append(v_current)

            v1, v2 = neighbours[v_current]
            if v1 != v_previous:
                v_current, v_previous = v1, v_current
            elif v2 != v_previous:
                v_current, v_previous = v2, v_current
            else:
                raise RuntimeError(f"Next vertices to visit ({v1=}, {v2=}) are both equal to {v_previous=}.")

        if close_paths:
            boundary_vertices.append(boundary_vertices[0])

        boundary_paths.append(mesh.vertices[boundary_vertices])

        boundary_edges = set(e for e in boundary_edges if e[0] not in boundary_vertices)
    return boundary_paths          
                
