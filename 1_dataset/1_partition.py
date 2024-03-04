import os
import xml.etree.ElementTree as ET
import numpy as np
from stl import mesh
import tqdm
import open3d as o3d
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
r=11
for dir_tmp in [dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8,dir9,dir10]:
    subfolders = [f.path for f in os.scandir(dir_tmp) if f.is_dir()]

    for patients , subfolder in tqdm.tqdm(enumerate(subfolders)):
        print('patient',patients,subfolder)
        if not os.path.exists(os.path.join(output,subfolder)):os.makedirs(os.path.join(output,subfolder))
        constructionInfo = [f.path for f in os.scandir(subfolder) if f.name.endswith('Info')]
        if len(constructionInfo)!=1:continue
        tree = ET.parse(constructionInfo[0])
        root = tree.getroot()
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            rotation_matrix_elements = tooth_element.findall('.//ZRotationMatrix')
            affine_matrix = np.zeros((4, 4))
            if len(rotation_matrix_elements)==0 : continue
            for i in range(4):
                for j in range(4):
                    element_name = f'_{i}{j}'
                    element_value = float(rotation_matrix_elements[0].find(element_name).text)
                    affine_matrix[i][j] = element_value
            affine_matrix = np.transpose(affine_matrix)
            inverse_affine_matrix = np.linalg.inv(affine_matrix)

            affine_margin = []
            vec3_elements = tooth_element.findall('.//Vec3')
            if tooth_element.findall('.//Margin') ==[]:
                continue
            for vec3_element in vec3_elements:
                x = float(vec3_element.find('x').text)
                y = float(vec3_element.find('y').text)
                z = float(vec3_element.find('z').text)
                original_margin = np.dot(inverse_affine_matrix,np.array([x, y, z, 1.0]))
                affine_margin.append(original_margin[:3])

            with open(output+'/'+subfolder+'/'+str(m)+'-margin'+'.obj', 'w') as obj_file:
                for margin in affine_margin:
                    obj_file.write(f'v {margin[0]} {margin[1]} {margin[2]}\n')

            center= tooth_element.find('.//Center')
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            affine_center = np.dot(inverse_affine_matrix,np.array([x, y, z, 1.0]))[:3]

            with open(output+'/'+subfolder+'/'+str(m)+'-center'+'.obj', 'w') as obj_file:
                obj_file.write(f'v {affine_center[0]} {affine_center[1]} {affine_center[2]}\n')

            patch_crop=[]
            face_crop=[]
            preparations_stl = tooth_element.find('ToothScanFileName').text
            mesh2 = mesh.Mesh.from_file(os.path.join(subfolder,preparations_stl))
            cout = 0
            for i in range(len(mesh2.points)):
                add_face=False
                if np.linalg.norm(mesh2.points[i][:3]-affine_center) <=r\
                and np.linalg.norm(mesh2.points[i][3:6]-affine_center) <=r\
                and np.linalg.norm(mesh2.points[i][6:9]-affine_center) <=r:
                    add_face= True
                if add_face:
                    patch_crop.append(mesh2.points[i])
                    face_crop.append(i)
                    cout=cout+1
            if cout ==0:
                print('no mesh around centers at ', str(m),subfolder)
                continue

            dict_xyz_index={}
            points=[]
            count = 1
            for index, vertices in enumerate(patch_crop):
                for xyz in [vertices[:3],vertices[3:6],vertices[6:]]:
                    vertices_str1 = ','.join([str(number) for number in xyz])
                    if vertices_str1 not in dict_xyz_index.keys():
                        dict_xyz_index[vertices_str1]= count
                        count= count+1
                        points.append(xyz)

            faces=[]
            for f , face in enumerate(patch_crop):
                one_face=[]
                for point in [face[:3],face[3:6],face[6:]]:
                    point_str = ','.join([str(number) for number in point])
                    point_index = dict_xyz_index[point_str]
                    one_face.append(point_index)
                faces.append(one_face)

            with open(output+'/'+subfolder+'/'+str(m)+'-patch_crop'+'.obj', 'w') as obj_file:
                for k, xyz in enumerate(points):
                    obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
                for k, face in enumerate(faces):
                    obj_file.write(f'f {face[0]} {face[1]} {face[2]}\n')

            open3d_mesh = o3d.geometry.TriangleMesh()
            open3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
            open3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))

            triangle_clusters, cluster_n_triangles, cluster_area = open3d_mesh.cluster_connected_triangles()
            faces2=[]
            index_max = np.max(np.array(cluster_n_triangles))
            index = cluster_n_triangles.index(index_max)
            for i , tc in enumerate(triangle_clusters):
                if tc==index:
                    faces2.append(open3d_mesh.triangles[i])

            point_distinct=[]
            for face in  faces2:
                for point in face:
                    if point not in point_distinct:
                        point_distinct.append(point)
            point_distinct.sort()

            points2=[]
            for point in point_distinct:
                points2.append(points[point-1])

            faces3=[]
            for face in faces2:
                point3=[]
                for point in face:
                    point3.append(point_distinct.index(point)+1)
                faces3.append(point3)

            with open(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj', 'w') as obj_file:
                for i, xyz in enumerate(points2):
                    obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
                for i, xyz in enumerate(faces3):
                    obj_file.write(f'f {xyz[0]} {xyz[1]} {xyz[2]}\n')
