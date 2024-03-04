import os
import xml.etree.ElementTree as ET
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
for dir_tmp in [dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8,dir9,dir10]:
    subfolders = [f.path for f in os.scandir(dir_tmp) if f.is_dir()]

    for patients , subfolder in tqdm.tqdm(enumerate(subfolders)):
        if not os.path.exists(os.path.join(output,subfolder)):os.makedirs(os.path.join(output,subfolder))
        constructionInfo = [f.path for f in os.scandir(subfolder) if f.name.endswith('Info')]

        if len(constructionInfo)!=1:
            continue
        tree = ET.parse(constructionInfo[0])
        root = tree.getroot()

        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):
            if tooth_element.findall('.//Margin') ==[]:
                continue
            if not os.path.exists(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj'):
                print(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj', 'not exist')
                continue
            open3d_mesh2 = o3d.io.read_triangle_mesh(output+'/'+subfolder+'/'+str(m)+'-partition'+'.obj')
            try :
                half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(open3d_mesh2)
            except:
                continue
            boundary_list = half_edge_mesh.get_boundaries()
            with open(output+'/'+subfolder+'/'+str(m)+'-boundary'+'.obj', 'w') as obj_file:
                for i in boundary_list[0]:
                    obj_file.write(f'v {open3d_mesh2.vertices[i][0]} {open3d_mesh2.vertices[i][1]} {open3d_mesh2.vertices[i][2]}\n')
