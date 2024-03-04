import os
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import trimesh
import h5py
def makedataset():
    test_dataset_h5 = 'output\\dataset_test.h5'
    with h5py.File(test_dataset_h5,'w') as hf:
        partition_labeled = 'output\\99-partition.obj'
        if os.path.exists(partition_labeled):

            i_mesh = trimesh.load(partition_labeled)


            cells = []
            normals = []
            centers = []

            for face in i_mesh.faces:
                face_vertices = i_mesh.vertices[face]
                face_normals = i_mesh.vertex_normals[face]

                cell = np.vstack(face_vertices).flatten()
                normal = np.vstack(face_normals).flatten()

                cells.append(cell)
                centers.append(np.mean(face_vertices, axis=0))
                normals.append(normal)

            cells = np.vstack(cells)
            normals = np.vstack(normals)
            centers = np.array(centers) / 3


            face_color_labels= [0]*len(i_mesh.visual.face_colors)
            face_color_labels=np.array(face_color_labels)
            grp = hf.create_group(partition_labeled)
            grp.create_dataset('cells',data=np.array(cells))
            grp.create_dataset('center',data=np.array(centers))
            grp.create_dataset('normals',data=np.array(normals))
            grp.create_dataset('face_normals',data=i_mesh.face_normals)
            grp.create_dataset('face_label',data=np.array(face_color_labels))

    print(f'Data has been successfully stored in {test_dataset_h5}.')
