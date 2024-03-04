from PyQt5 import QtCore, QtGui, QtWidgets
import trimesh
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkIdTypeArray
from vtkmodules.vtkCommonDataModel import (
    vtkSelection,
    vtkSelectionNode,
    vtkUnstructuredGrid
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersExtraction import vtkExtractSelection
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkDataSetMapper
)
from vtkmodules.vtkIOGeometry import (
    vtkOBJReader,
    vtkSTLReader
)

import vtk
import sys
from PyQt5 import QtCore
from vtk import *
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import Qt

import os
import xml.etree.ElementTree as ET
import numpy as np
from stl import mesh
import open3d as o3d
from trimesh import proximity
# Catch mouse events
selected_point = True
selected_point_xyz = []
class MouseInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, data):
        self.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.data = data
        self.selected_mapper = vtkDataSetMapper()
        self.selected_actor = vtkActor()

    def left_button_press_event(self, obj, event):
        colors = vtkNamedColors()

        # Get the location of the click (in window coordinates)
        pos = self.GetInteractor().GetEventPosition()

        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)

        # Pick from this location.
        picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())

        world_position = picker.GetPickPosition()
        print(selected_point )
        print(f'Cell id is: {picker.GetCellId()}')


        if picker.GetCellId() != -1:
            global selected_point_xyz
            selected_point_xyz =[ i for i in  world_position]
            print(f'Pick position is: ({world_position[0]:.6g}, {world_position[1]:.6g}, {world_position[2]:.6g})')

            ids = vtkIdTypeArray()
            ids.SetNumberOfComponents(1)
            ids.InsertNextValue(picker.GetCellId())

            selection_node = vtkSelectionNode()
            selection_node.SetFieldType(vtkSelectionNode.CELL)
            selection_node.SetContentType(vtkSelectionNode.INDICES)
            selection_node.SetSelectionList(ids)

            selection = vtkSelection()
            selection.AddNode(selection_node)

            extract_selection = vtkExtractSelection()
            extract_selection.SetInputData(0, self.data)
            extract_selection.SetInputData(1, selection)
            extract_selection.Update()

            # In selection
            selected = vtkUnstructuredGrid()
            selected.ShallowCopy(extract_selection.GetOutput())

            print(f'Number of points in the selection: {selected.GetNumberOfPoints()}')
            print(f'Number of cells in the selection : {selected.GetNumberOfCells()}')

            self.selected_mapper.SetInputData(selected)
            self.selected_actor.SetMapper(self.selected_mapper)
            self.selected_actor.GetProperty().EdgeVisibilityOn()
            self.selected_actor.GetProperty().SetColor(colors.GetColor3d('Tomato'))

            self.selected_actor.GetProperty().SetLineWidth(3)

            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.selected_actor)

        # Forward events
        self.OnLeftButtonDown()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # horizontalLayout
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # buttonContainer
        self.buttonContainer = QtWidgets.QWidget(self.centralwidget)
        self.buttonContainer.setObjectName("buttonContainer")

        # verticalButtonLayout
        self.verticalButtonLayout = QtWidgets.QVBoxLayout(self.buttonContainer)
        self.verticalButtonLayout.setObjectName("verticalButtonLayout")

        # pushButton load
        self.pushButton = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton.setObjectName("pushButton")
        self.verticalButtonLayout.addWidget(self.pushButton)

        # pushButton1 show all
        self.pushButton1 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton1.setObjectName("pushButton1")
        self.verticalButtonLayout.addWidget(self.pushButton1)

        # pushButton2 hide
        self.pushButton2 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton2.setObjectName("pushButton2")
        self.verticalButtonLayout.addWidget(self.pushButton2)


        # pushButton4 get patch gt
        self.pushButton4 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton4.setObjectName("pushButton4")
        self.verticalButtonLayout.addWidget(self.pushButton4)

        # pushButton5 get patition gt
        self.pushButton5 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton5.setObjectName("pushButton5")
        self.verticalButtonLayout.addWidget(self.pushButton5)

        # pushButton3 get margin gt
        self.pushButton3 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton3.setObjectName("pushButton3")
        self.verticalButtonLayout.addWidget(self.pushButton3)

        # pushButton6 labeler gt
        self.pushButton6 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton6.setObjectName("pushButton6")
        self.verticalButtonLayout.addWidget(self.pushButton6)

        # pushButton7 get margin prediction
        self.pushButton7 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton7.setObjectName("pushButton7")
        self.verticalButtonLayout.addWidget(self.pushButton7)

        # pushButton8 get margin prediction
        self.pushButton8 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton8.setObjectName("pushButton8")
        self.verticalButtonLayout.addWidget(self.pushButton8)

        # pushButton9 get margin prediction
        self.pushButton9 = QtWidgets.QPushButton(self.buttonContainer)
        self.pushButton9.setObjectName("pushButton9")
        self.verticalButtonLayout.addWidget(self.pushButton9)


        # buttonContainer to horizontalLayout
        self.horizontalLayout.addWidget(self.buttonContainer)


        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout.addWidget(self.frame)

        self.horizontalLayout.setStretchFactor(self.frame, 2)
        self.horizontalLayout.setStretchFactor(self.buttonContainer, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load"))
        self.pushButton1.setText(_translate("MainWindow", "Show all "))
        self.pushButton2.setText(_translate("MainWindow", "Hide all "))
        self.pushButton4.setText(_translate("MainWindow", "Get patch GT"))
        self.pushButton5.setText(_translate("MainWindow", "Get partition GT"))
        self.pushButton3.setText(_translate("MainWindow", "Get margin GT"))
        self.pushButton6.setText(_translate("MainWindow", "labeler GT"))
        self.pushButton7.setText(_translate("MainWindow", "Get patch"))
        self.pushButton8.setText(_translate("MainWindow", "Get margin Prediction"))
        self.pushButton9.setText(_translate("MainWindow", "Smoothing margin line"))
    def hideAllActors(self):
        actors = self.ren.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            actor.VisibilityOff()
            actor = actors.GetNextItem()
        self.iren.Render()
    def showAllActors(self):
        actors = self.ren.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            actor.VisibilityOn()
            actor = actors.GetNextItem()
        self.iren.Render()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl = Qt.QVBoxLayout(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()


        self.style = MouseInteractorStyle(None)
        self.style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(self.style)

        self.pushButton.clicked.connect(self.OpenVTK)
        self.pushButton1.clicked.connect(self.showAllActors)
        self.pushButton2.clicked.connect(self.hideAllActors)
        self.pushButton3.clicked.connect(self.get_margin_gt)
        self.pushButton4.clicked.connect(self.get_patch)
        self.pushButton5.clicked.connect(self.get_partition)
        self.pushButton6.clicked.connect(self.labeler)
        self.pushButton7.clicked.connect(self.get_patch2)
        self.pushButton8.clicked.connect(self.get_margin_prediction)
        self.pushButton9.clicked.connect(self.smoothing_margin_line)
    def smoothing_margin_line(self):
        from interpolated import interpolated
        from smoothing_gauss import smmothing
        from biou_open3d import get_boundary
        get_boundary()
        interpolated()
        smmothing()
        self.hideAllActors()
        self.loadSTL('output\\99-partition.obj')
        mesh_ = trimesh.load ('output\\99-snake_smoothing10_proximity.obj')
        self.loadSTL('output\\99-snake_smoothing10_proximity.obj',point_size = 0.2 , face_color = np.array([[255,0,0]*len(mesh_.vertices)]))


    def get_margin_prediction(self):
        from testing import test
        from dataset_test import makedataset
        from labeler import labeler2
        makedataset()
        test()
        labeler2()
        self.hideAllActors()
        mesh_ = trimesh.load ('output\\99-partition-prediction.obj')
        self.loadSTL('output\\99-partition-prediction.obj',face_color=mesh_.visual.vertex_colors )

    def get_patch2(self):
        subfolder = ".\\2022-01-07_00001-001"
        constructionInfo = ".\\2022-01-07_00001-001\\2022-01-07_00001-001.constructionInfo"
        output = 'output'
        tree = ET.parse(constructionInfo)
        root = tree.getroot()
        tooth = root.findall('.//Tooth')

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        tooth_element  = tooth_elements[0]

        selected_point_xyz[2] -= 7
        affine_center = selected_point_xyz
        r=11
        m=99
        ############################################################################
        #get patch crop
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
        #vertices distinct
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


        #divise by margin

        open3d_mesh = o3d.geometry.TriangleMesh()
        open3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        open3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))


        triangle_clusters, cluster_n_triangles, cluster_area = open3d_mesh.cluster_connected_triangles()
        index_max = np.argmax(cluster_n_triangles)
        faces2 = [open3d_mesh.triangles[i] for i, tc in enumerate(triangle_clusters) if tc == index_max]


        point_distinct = list(set(point for face in faces2 for point in face))
        point_distinct.sort()


        points2 = [points[point - 1] for point in point_distinct]


        point_index_mapping = {point: index + 1 for index, point in enumerate(point_distinct)}
        faces3 = [[point_index_mapping[point] for point in face] for face in faces2]


        with open(output+'/'+str(m)+'-partition'+'.obj', 'w') as obj_file:
            for i, xyz in enumerate(points2):
                obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
            for i, xyz in enumerate(faces3):
                obj_file.write(f'f {xyz[0]} {xyz[1]} {xyz[2]}\n')

        print('get partition ')
        self.hideAllActors()
        self.loadSTL(output+'/'+str(m)+'-partition'+'.obj')
        print('show partition ')
        return
    def OpenVTK(self):
        self.loadSTL(".\\2022-01-07_00001-001\\2022-01-07_00001-001-LowerJaw.stl")

    def Coping(self):
        self.loadSTL(".\\2022-01-07_00001-001\\2022-01-07_00001-001-UpperJaw.stl")

    def loadSTL(self, filename,point_size=False,face_color=np.array([])):

        if filename.endswith('stl'):
            reader = vtkSTLReader()
        else:
            reader = vtkOBJReader()
        reader.SetFileName(filename)

        actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        if face_color.any():
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")

            for i in range(face_color.shape[0]):
                colors.InsertNextTuple(face_color[i][:3])
            mapper.Update()
            mapper.GetInput().GetPointData().SetScalars(colors)
        else:
            #print("Warning: Invalid color format. Please provide a NumPy array of shape (1000, 3).")
            actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('grey'))


        # crate actor

        actor.SetMapper(mapper)


        # add actor
        self.ren.AddActor(actor)

        ###
        if point_size :
            tri_mesh = trimesh.load(filename)
            for point in tri_mesh.vertices:
                # Create a new sphere for each point
                sphere = vtkSphereSource()
                sphere.SetRadius(point_size)  # You can adjust the radius as needed
                sphere.SetCenter(point)

                sphere_mapper = vtk.vtkPolyDataMapper()
                sphere_mapper.SetInputConnection(sphere.GetOutputPort())

                sphere_actor = vtk.vtkActor()
                sphere_actor.GetProperty().SetColor(vtkNamedColors().GetColor3d('Red'))
                sphere_actor.SetMapper(sphere_mapper)

                # Add each sphere actor to the renderer
                self.ren.AddActor(sphere_actor)
        ####

        self.ren.SetBackground(vtkNamedColors().GetColor3d('PaleTurquoise'))
        self.ren.ResetCamera()

        # style
        style = MouseInteractorStyle(reader.GetOutput())
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        # update
        self.iren.Initialize()
        self.iren.Start()


    def get_margin_gt(self):
        self.hideAllActors()
        size=0.2
        constructionInfo = ".\\2022-01-07_00001-001\\2022-01-07_00001-001.constructionInfo"
        output = 'output'
        tree = ET.parse(constructionInfo)
        root = tree.getroot()
        tooth = root.findall('.//Tooth')

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            if os.path.exists(output+'/'+str(m)+'-center'+'.obj') and \
                os.path.exists(output+'/'+str(m)+'-margin'+'.obj'):
                self.loadSTL(output+'/'+str(m)+'-margin'+'.obj',size)
                self.loadSTL(output+'/'+str(m)+'-center'+'.obj',size)
                continue

            #get affine metrix
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


            with open(output+'/'+str(m)+'-margin'+'.obj', 'w') as obj_file:
                for margin in affine_margin:
                    obj_file.write(f'v {margin[0]} {margin[1]} {margin[2]}\n')
            print('get margin ')
            self.loadSTL(output+'/'+str(m)+'-margin'+'.obj',size)
            print('show margin ')
            #get affined center
            center= tooth_element.find('.//Center')
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            affine_center = np.dot(inverse_affine_matrix,np.array([x, y, z, 1.0]))[:3]

            with open(output+'/'+str(m)+'-center'+'.obj', 'w') as obj_file:
                obj_file.write(f'v {affine_center[0]} {affine_center[1]} {affine_center[2]}\n')
            print('get center ')
            self.loadSTL(output+'/'+str(m)+'-center'+'.obj',size)
            print('show center ')
        return

    def get_patch(self):
        self.hideAllActors()
        r =11
        subfolder = ".\\2022-01-07_00001-001"
        constructionInfo = ".\\2022-01-07_00001-001\\2022-01-07_00001-001.constructionInfo"
        output = 'output'
        tree = ET.parse(constructionInfo)
        root = tree.getroot()
        tooth = root.findall('.//Tooth')

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            if os.path.exists(output+'/'+str(m)+'-patch_crop'+'.obj') :
                self.loadSTL(output+'/'+str(m)+'-patch_crop'+'.obj')
                continue

            #get affine metrix
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

            #get affined center
            center= tooth_element.find('.//Center')
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            affine_center = np.dot(inverse_affine_matrix,np.array([x, y, z, 1.0]))[:3]
            ############################################################################
            #get patch crop
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
            #vertices distinct
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

            with open(output+'/'+str(m)+'-patch_crop'+'.obj', 'w') as obj_file:
                for k, xyz in enumerate(points):
                    obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
                for k, face in enumerate(faces):
                    obj_file.write(f'f {face[0]} {face[1]} {face[2]}\n')
            print('get patch ')
            self.loadSTL(output+'/'+str(m)+'-patch_crop'+'.obj')
            print('show patch ')
        return

    def get_partition(self):
        self.hideAllActors()
        r =11
        subfolder = ".\\2022-01-07_00001-001"
        constructionInfo = ".\\2022-01-07_00001-001\\2022-01-07_00001-001.constructionInfo"
        output = 'output'
        tree = ET.parse(constructionInfo)
        root = tree.getroot()
        tooth = root.findall('.//Tooth')

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            if os.path.exists(output+'/'+str(m)+'-partition'+'.obj') :
                self.loadSTL(output+'/'+str(m)+'-partition'+'.obj')
                continue

            #get affine metrix
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

            #get affined center
            center= tooth_element.find('.//Center')
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            affine_center = np.dot(inverse_affine_matrix,np.array([x, y, z, 1.0]))[:3]
            ############################################################################
            #get patch crop
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
            #vertices distinct
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


            #divise by margin

            open3d_mesh = o3d.geometry.TriangleMesh()
            open3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
            open3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))


            triangle_clusters, cluster_n_triangles, cluster_area = open3d_mesh.cluster_connected_triangles()
            index_max = np.argmax(cluster_n_triangles)
            faces2 = [open3d_mesh.triangles[i] for i, tc in enumerate(triangle_clusters) if tc == index_max]


            point_distinct = list(set(point for face in faces2 for point in face))
            point_distinct.sort()


            points2 = [points[point - 1] for point in point_distinct]


            point_index_mapping = {point: index + 1 for index, point in enumerate(point_distinct)}
            faces3 = [[point_index_mapping[point] for point in face] for face in faces2]


            with open(output+'/'+str(m)+'-partition'+'.obj', 'w') as obj_file:
                for i, xyz in enumerate(points2):
                    obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
                for i, xyz in enumerate(faces3):
                    obj_file.write(f'f {xyz[0]} {xyz[1]} {xyz[2]}\n')

            print('get partition ')
            self.loadSTL(output+'/'+str(m)+'-partition'+'.obj')
            print('show partition ')

        return

    def labeler(self):
        self.hideAllActors()
        r =11
        subfolder = ".\\2022-01-07_00001-001"
        constructionInfo = ".\\2022-01-07_00001-001\\2022-01-07_00001-001.constructionInfo"
        output = 'output'
        tree = ET.parse(constructionInfo)
        root = tree.getroot()
        tooth = root.findall('.//Tooth')

        #get affined margin
        tooth_elements = root.findall('.//Tooth')
        for m,tooth_element  in enumerate(tooth_elements):

            if tooth_element.findall('.//Margin') ==[]:
                continue
            if not os.path.exists(output+'/'+str(m)+'-partition'+'.obj'):
                print(output+'/'+str(m)+'-partition'+'.obj', 'not exist')
                continue
            open3d_mesh2 = o3d.io.read_triangle_mesh(output+'/'+str(m)+'-partition'+'.obj')
            try :
                half_edge_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(open3d_mesh2)
            except:
                continue
            boundary_list = half_edge_mesh.get_boundaries()

            with open(output+'/'+str(m)+'-boundary'+'.obj', 'w') as obj_file:
                for i in boundary_list[0]:
                    obj_file.write(f'v {open3d_mesh2.vertices[i][0]} {open3d_mesh2.vertices[i][1]} {open3d_mesh2.vertices[i][2]}\n')


            partition = trimesh.load(output+'/'+str(m)+'-partition'+'.obj')

            if not os.path.exists((output+'/'+str(m)+'-boundary'+'.obj')):
                continue

            boundary = trimesh.load(output+'/'+str(m)+'-boundary'+'.obj')
            margin = trimesh.load(output+'/'+str(m)+'-margin'+'.obj')


            tid = proximity.nearby_faces(partition, margin.vertices)

            tid_distinct = list(set(i for j in tid for i in j))

            points_color = {
                point_str: '1.00000000 0.00000000 0.00000000'
                for i in tid_distinct
                for pfi in partition.faces[i]
                for point_str in [','.join(map(str, partition.vertices[pfi]))]
            }


            face_copy = partition.faces
            arr = np.delete(partition.faces, tid_distinct, axis=0)
            partition.faces = arr


            # face cluster
            open3d_mesh = o3d.geometry.TriangleMesh()
            open3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(partition.vertices))
            open3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(partition.faces))
            triangle_clusters, cluster_n_triangles, cluster_area = open3d_mesh.cluster_connected_triangles()


            label1 = [open3d_mesh.triangles[i] for i, tc in enumerate(triangle_clusters) if tc == 0]
            label2 = [open3d_mesh.triangles[i] for i, tc in enumerate(triangle_clusters) if tc == 1]


            bv_key_set = set(','.join(map(str, bv)) for bv in boundary.vertices)
            bv_key = list(bv_key_set)



            labels = np.zeros(len(partition.vertices))
            partie1_set = set(','.join(map(str, partition.vertices[face])) for faces in label1 for face in faces)
            partie1 = list(partie1_set)


            partie2_set = set(','.join(map(str, partition.vertices[face])) for faces in label2 for face in faces)
            partie2 = list(partie2_set)

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


            with open(output+'/'+str(m)+'-partition-labeled'+'.obj', 'w') as obj_file:
                for i, xyz in enumerate(partition.vertices):
                    color_index = ','.join([str(number) for number in xyz])
                    if color_index in points_color.keys():
                        obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]} {points_color[color_index]}\n')
                    else:
                        obj_file.write(f'v {xyz[0]} {xyz[1]} {xyz[2]} 1.00000000 1.00000000 1.00000000 \n')
                for i, xyz in enumerate(face_copy):
                    obj_file.write(f'f {xyz[0]+1} {xyz[1]+1} {xyz[2]+1}\n')

            print('get labeler ')
            tri_mesh = trimesh.load(output+'/'+str(m)+'-partition-labeled'+'.obj')
            self.loadSTL(output+'/'+str(m)+'-partition-labeled'+'.obj',face_color=tri_mesh.visual.vertex_colors)
            print('show labeler ')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
