import os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import h5py
dir1='1-8-23'
dir2='1-9-8'
dir3='1-10-15'
dir4='1-11-40'
dir5='1-12-18'
dir6='1-13-32'
dir7='1-14-21'
dir8='1-15-2'
dir9='12-21-38'
dir10='12-22-30'

output='data'
train_dataset_h5 = '.\\marginSeg\\predict\\data\\dataset_train.h5'
test_dataset_h5 = '.\\marginSeg\\predict\\data\\dataset_test.h5'
train_list=[]
test_list=[]
with h5py.File(test_dataset_h5,'w') as hf:
    for dir_tmp in [dir1]:
        c=0
        subfolders = [f.path for f in os.scandir(dir_tmp) if f.is_dir()]
        print("Subfolders in the specified folder:")
        for patients , subfolder in enumerate(subfolders):
            if not os.path.exists(os.path.join(output,subfolder)):os.makedirs(os.path.join(output,subfolder))
            constructionInfo = [f.path for f in os.scandir(subfolder) if f.name.endswith('Info')]
            if len(constructionInfo)!=1:continue
            tree = ET.parse(constructionInfo[0])
            root = tree.getroot()
            tooth_elements = root.findall('.//Tooth')
            for m,tooth_element  in enumerate(tooth_elements):
                partition_labeled = os.path.join(output,subfolder,str(m)+'-partition-labeled.obj')
                if os.path.exists(partition_labeled):
                    print(partition_labeled)
                    i_mesh = trimesh.load(partition_labeled)
                    cells=[]
                    normals=[]
                    centers=[]
                    for face in i_mesh.faces:
                        cell=[]
                        normal=[]
                        center = [0,0,0]
                        for point_index in face :
                            cell.append(i_mesh.vertices[point_index])
                            center += i_mesh.vertices[point_index]
                            normal.append(i_mesh.vertex_normals[point_index])
                        cell = np.array(cell).reshape(-1)
                        normal = np.array(normal).reshape(-1)
                        cells.append(cell)
                        centers.append(center/3)
                        normals.append(normal)
                    face_color_labels= []
                    for j, fc in enumerate(i_mesh.visual.face_colors):
                        if np.array_equal(fc, [0,0,255,255]):
                            face_color_labels.append(1)
                        else:
                            face_color_labels.append(0)
                    face_color_labels=np.array(face_color_labels)
                    grp = hf.create_group(partition_labeled)
                    grp.create_dataset('cells',data=np.array(cells))
                    grp.create_dataset('center',data=np.array(centers))
                    grp.create_dataset('normals',data=np.array(normals))
                    grp.create_dataset('face_normals',data=i_mesh.face_normals)
                    grp.create_dataset('face_label',data=np.array(face_color_labels))
print(f'Data has been successfully stored in {test_dataset_h5}.')
