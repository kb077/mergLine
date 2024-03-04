import os
import xml.etree.ElementTree as ET
import numpy as np
import tqdm
import open3d as o3d
import trimesh
from trimesh import proximity
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
for dir_tmp in [dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8,dir9,dir10]:
    subfolders = [f.path for f in os.scandir(dir_tmp) if f.is_dir()]

    print("Subfolders in the specified folder:")
    for patients , subfolder in tqdm.tqdm(enumerate(subfolders)):
        print(subfolder)
        if not os.path.exists(os.path.join(output,subfolder)):os.makedirs(os.path.join(output,subfolder))

        constructionInfo = [f.path for f in os.scandir(subfolder) if f.name.endswith('Info')]

        if len(constructionInfo)!=1:continue
        tree = ET.parse(constructionInfo[0])
        root = tree.getroot()

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            if tooth_element.findall('.//Margin') ==[]:
                continue
            if not os.path.exists(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj'):
                print(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj', 'not exist')
                continue
            partition = trimesh.load(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj')

            if not os.path.exists((output+'/'+subfolder+'/'+str(m)+'-boundary'+'.obj')):
                continue

            boundary = trimesh.load(output+'/'+subfolder+'/'+str(m)+'-boundary'+'.obj')
            margin = trimesh.load(output+'/'+subfolder+'/'+str(m)+'-margin'+'.obj')

            #delete nearby face
            tid  = proximity.nearby_faces(partition,margin.vertices)
            c=0
            tid_disctint=[]
            points_color={}
            for j in tid:
                for i in j:
                    if i not in tid_disctint:
                        tid_disctint.append(i)
                        for pfi in partition.faces[i]:
                            point_str =  ','.join([str(number) for number in partition.vertices[pfi]])
                            points_color[point_str] = '1.00000000 0.00000000 0.00000000'

            face_copy = partition.faces
            arr = np.copy(partition.faces)
            tid_disctint.sort()
            for i in tid_disctint:
                arr = np.delete(arr, i-c, 0)
                c= c+1
            partition.faces= arr

            # face cluster
            open3d_mesh = o3d.geometry.TriangleMesh()
            open3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(partition.vertices))
            open3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(partition.faces))
            triangle_clusters, cluster_n_triangles, cluster_area = open3d_mesh.cluster_connected_triangles()

            label1,label2,label3=[],[],[]
            for i , tc in enumerate(triangle_clusters):
                if tc==0:
                    label1.append(open3d_mesh.triangles[i])
                if tc==1:
                    label2.append(open3d_mesh.triangles[i])

            #set color
            bv_key= []
            for bv in boundary.vertices:
                bv_str = ','.join([str(number) for number in bv])
                if bv_str not in bv_key:
                    bv_key.append(bv_str)

            labels= np.zeros(len(partition.vertices))
            partie1=[]
            for faces in label1:
                for face in faces:
                    point_str = ','.join([str(number) for number in partition.vertices[face]])
                    if point_str not in partie1:
                        partie1.append(point_str)

            partie2=[]
            for faces in label2:
                for face in faces:
                    point_str = ','.join([str(number) for number in partition.vertices[face]])
                    if point_str not in partie2:
                        partie2.append(point_str)
            #count intersection

            bv_key = np.array(bv_key)
            interse_p1=len(bv_key[np.isin(bv_key, partie1)])
            interse_p2=len(bv_key[np.isin(bv_key, partie2)])
            #
            if interse_p1 >interse_p2:
                for p1 in partie1:
                    points_color[p1]= '0.00000000 1.00000000 0.00000000'
                for p2 in partie2:
                    points_color[p2]= '0.00000000 0.00000000 1.00000000'
            else:
                for p1 in partie1:
                    points_color[p1]= '0.00000000 0.00000000 1.00000000'
                for p2 in partie2:
                    points_color[p2]= '0.00000000 1.00000000 0.00000000'

            with open(output+'/'+subfolder+'/'+str(m)+'-partition-labeled'+'.obj', 'w') as obj_file:
                for i, xyz in enumerate(partition.vertices):
                    color_index = ','.join([str(number) for number in xyz])
                    if color_index in points_color.keys():
                        obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]} {points_color[color_index]}\n')
                    else:
                        obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]} 1.00000000 1.00000000 1.00000000 \n')
                for i, xyz in enumerate(face_copy):
                    obj_file.write(f'f {xyz[0]+1} {xyz[1]+1} {xyz[2]+1}\n')
