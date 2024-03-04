import trimesh
import numpy as np
def get_boundary():
    model_name ='pointmlp-full-dataset-biou-alpha'
    path= 'output'
    j=99
    color_path = path+'\\'+str(j)+'-partition-prediction.obj'

    mesh2 = trimesh.load(color_path)
    meshcolor2 = np.array(mesh2.visual.face_colors) #(33867, 4)
    selected_indices = np.where((meshcolor2[:, 2] !=0))[0]
    sub_mesh2 = mesh2.submesh([selected_indices])
    sub_mesh2[0].export(path+'\\'+str(j)+'-'+model_name+'-prediction-cut.obj')
    return 

