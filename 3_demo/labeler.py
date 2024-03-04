import trimesh
import numpy as np
def labeler2():
    j=99
    mesh = trimesh.load('output\\'+str(j)+'-partition.obj')
    color = np.load('output\\'+str(j)+'-partition.obj-labels-pointmlp-full-dataset-biou-alpha.npy')
    color = color.reshape(len(color),1)
    face_color = []
    for c in color :
        if c==1:
            face_color.append([0,0,255,255])
        elif c==0:
            face_color.append([0,255,0,255])
        else:
            face_color.append([0,0,0,255])
    mesh.visual.face_colors = face_color
    mesh.export('output\\'+str(j)+'-partition-prediction.obj')
