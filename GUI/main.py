# This Python file uses the following encoding: utf-8
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'

from PySide2 import QtWidgets, QtCore
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtCore import QFile
from PySide2 import QtGui
from PySide2.QtUiTools import QUiLoader


from pvtrace import *
#from pvtrace.geometry.utils import EPS_ZERO
from pvtrace.light.utils import wavelength_to_rgb
from pvtrace.material.utils import lambertian
from pvtrace.light.event import Event
import time
import functools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import trimesh
import pandas as pd
from dataclasses import asdict
import progressbar 
import trimesh
from matplotlib.pyplot import plot, hist, scatter
import json


class testingQT(QWidget):
    def __init__(self):
        super(testingQT, self).__init__()
        main_widget = self.load_ui()
        
        # layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(main_widget)
        # self.setLayout(layout)
        
        self.inputShape = self.findChild(QtWidgets.QComboBox,'comboBox')
        self.STLfile = ''
        self.lumophore = self.findChild(QtWidgets.QComboBox,'comboBox_2')
        self.lumophoreConc = self.findChild(QtWidgets.QLineEdit,'lineEdit_20')
        self.waveguideAbs = self.findChild(QtWidgets.QLineEdit,'lineEdit_23')
        self.waveguideN = self.findChild(QtWidgets.QLineEdit,'lineEdit_24')
        self.lumophorePLQY = self.findChild(QtWidgets.QLineEdit,'lineEdit_25')
        self.dimx = self.findChild(QtWidgets.QLineEdit,'lineEdit')
        self.dimy = self.findChild(QtWidgets.QLineEdit,'lineEdit_13')
        self.dimz = self.findChild(QtWidgets.QLineEdit,'lineEdit_2')
        self.STLfileShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit')
        self.LumfileShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_4')
        self.enclosingBox = self.findChild(QtWidgets.QCheckBox,'checkBox_20')
        self.LSCbounds = np.array([])
        
        self.solarFaceAll = self.findChild(QtWidgets.QCheckBox, 'checkBox')
        self.solarFaceLeft = self.findChild(QtWidgets.QCheckBox,'checkBox_3')
        self.solarFaceRight = self.findChild(QtWidgets.QCheckBox,'checkBox_4')
        self.solarFaceFront = self.findChild(QtWidgets.QCheckBox,'checkBox_5')
        self.solarFaceBack = self.findChild(QtWidgets.QCheckBox,'checkBox_6')
        self.bottomMir = self.findChild(QtWidgets.QCheckBox,'checkBox_2')
        self.bottomScat = self.findChild(QtWidgets.QCheckBox,'checkBox_19')

        self.thinFilm = self.findChild(QtWidgets.QCheckBox,'checkBox_7')
        self.thinFilmThickness = self.findChild(QtWidgets.QLineEdit,'lineEdit_14')
    
        self.LSClayers = self.findChild(QtWidgets.QTabWidget,'tabWidget')
        
        self.lightPattern = self.findChild(QtWidgets.QComboBox,'comboBox_3')
        self.lightDimx = self.findChild(QtWidgets.QLineEdit,'lineEdit_4')
        self.lightDimy = self.findChild(QtWidgets.QLineEdit,'lineEdit_3')
        self.lightWavMin = self.findChild(QtWidgets.QLineEdit,'lineEdit_5')
        self.lightWavMax = self.findChild(QtWidgets.QLineEdit,'lineEdit_6')
        self.lightDiv = self.findChild(QtWidgets.QLineEdit,'lineEdit_7')
        
        self.numRays = self.findChild(QtWidgets.QLineEdit,'lineEdit_8')
        self.wavMin = self.findChild(QtWidgets.QLineEdit,'lineEdit_9')
        self.wavMax = self.findChild(QtWidgets.QLineEdit,'lineEdit_10')
        self.convPlot = self.findChild(QtWidgets.QCheckBox,'checkBox_21')
        self.convThres = self.findChild(QtWidgets.QLineEdit,'lineEdit_21')
        self.showSim = self.findChild(QtWidgets.QCheckBox,'checkBox_22')
        
        self.setSaveFolder = self.findChild(QtWidgets.QToolButton,'toolButton')
        self.saveFolder = ''
        self.saveFolderShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_2')
        # self.saveFileNameShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_3')
        self.saveFileNameShow = self.findChild(QtWidgets.QLineEdit, 'lineEdit_12')
        self.saveFileName = ''
        self.figDPI = self.findChild(QtWidgets.QLineEdit,'lineEdit_11')
        self.saveInputs = self.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.saveInputsFile = self.findChild(QtWidgets.QLineEdit,'lineEdit_22')
        self.loadInputs = self.findChild(QtWidgets.QPushButton, 'pushButton_3')

        
        
        # to do
        # absorption/scattering of waveguide
        # backscattering layer
        # add surface scattering to waveguide
        # add additional layered geometries
        
        self.inputShape.currentTextChanged.connect(self.onShapeChanged)
        self.rotateX = False
        self.rotateY = False
        
        self.solarFaceAll.stateChanged.connect(self.onSolarFaceAll)
    
        
        self.lightWavMin.textChanged.connect(self.onLightWavMinChanged)
        self.lightWavMax.textChanged.connect(self.onLightWavMaxChanged)
        
        self.dimx.textChanged.connect(self.onDimXChanged)
        self.dimy.textChanged.connect(self.onDimYChanged)
        
        self.setSaveFolder.clicked.connect(self.onSetSaveFolder)
        
        self.saveInputs.clicked.connect(self.onSaveInputs)
        self.loadInputs.clicked.connect(self.onLoadInputs)
        
        self.finishInput = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.finishInput.clicked.connect(self.onFinishInputClicked)
        self.thinFilm.clicked.connect(self.onThinFilmClicked)
        
        
        

    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(os.path.dirname(__file__), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        ui = loader.load(ui_file, self)
        ui_file.close()
        
        return ui
    
    def onDimXChanged(self):
        self.lightDimx.setText(self.dimx.text())
        if(self.inputShape.currentText() != 'Import Mesh'):
            self.dimy.setText(self.dimx.text())
            self.lightDimy.setText(self.dimx.text())
    
    def onDimYChanged(self):
        self.lightDimy.setText(self.dimy.text())
        
    def onSolarFaceAll(self):
        allEnabled = self.solarFaceAll.isChecked()
        self.solarFaceLeft.setChecked(allEnabled)
        self.solarFaceRight.setChecked(allEnabled)
        self.solarFaceFront.setChecked(allEnabled)
        self.solarFaceBack.setChecked(allEnabled)
    
    def onThinFilmClicked(self):
        if(self.thinFilm.isChecked()):
            self.thinFilmThickness.setEnabled(True)
        else:
            self.thinFilmThickness.setEnabled(False)
    
    def onLightWavMinChanged(self):
        pass
        # self.wavMin.setText(str(float(self.lightWavMin.text()) - 50))
        self.lightWavMax.setText(str(float(self.lightWavMin.text()) + 2))
        
    def onLightWavMaxChanged(self):
        pass
        # self.wavMax.setText(str(float(self.lightWavMax.text()) + 50))
    
    def onShapeChanged(self):
        if(self.inputShape.currentText() == 'Import Mesh'):
            self.STLfile = QtWidgets.QFileDialog.getOpenFileName(self, 'OpenFile')
            self.STLfile = self.STLfile[0]
            self.STLfileShow.setPlainText(self.STLfile)
            self.mesh = trimesh.load(self.STLfile)
            mesh = self.mesh
            self.LSCdims = mesh.extents
            LSCdims = self.LSCdims
            self.LSCbounds = mesh.bounds
            self.LSCbounds = self.LSCbounds - self.mesh.centroid
            self.dimx.setText("{:.3f}".format(LSCdims[0]))
            self.dimy.setText("{:.3f}".format(LSCdims[1]))
            self.dimz.setText("{:.3f}".format(LSCdims[2]))
            time.sleep(1)
            if(float(self.dimx.text()) < float(self.dimz.text())):
                self.rotateY = True
                self.dimx.setText("{:.3f}".format(LSCdims[2]))
                self.dimz.setText("{:.3f}".format(LSCdims[0]))
            elif(float(self.dimy.text()) < float(self.dimz.text())):
                self.rotateX = True
                self.dimy.setText("{:.3f}".format(LSCdims[2]))
                self.dimz.setText("{:.3f}".format(LSCdims[1]))
        else:
            self.STLfileShow.setPlainText('')
            
    def onSetSaveFolder(self):
        self.saveFolder = QtWidgets.QFileDialog.getExistingDirectory(self, 'OpenFile')
        # self.saveFolder = self.saveFolder[0]
        self.saveFolderShow.setPlainText(self.saveFolder)
        
    def onSaveInputs(self):
        data= {}
        data['LSC'] = []
        data['LSC'].append({
            'shape': self.inputShape.currentText(),
            'STLfile': self.STLfile,
            'dimX': (self.dimx.text()),
            'dimY': (self.dimy.text()),
            'dimZ': (self.dimz.text()),
            'PVedgesLRFB': [self.solarFaceLeft.isChecked(), self.solarFaceRight.isChecked(), self.solarFaceFront.isChecked(), self.solarFaceBack.isChecked()],
            'bottomMir': self.bottomMir.isChecked(),
            'bottomScat': self.bottomScat.isChecked(),
            'thinFilm': self.thinFilm.isChecked(),
            'lumophore': self.lumophore.currentText(),
            'lumophoreConc': (self.lumophoreConc.text()),
            'waveguideAbs': (self.waveguideAbs.text()),
            'lightPattern': self.lightPattern.currentText(),
            'lightDimX': (self.lightDimx.text()),
            'lightDimY': (self.lightDimy.text()),
            'lightWavMin': (self.lightWavMin.text()),
            'lightWavMax': (self.lightWavMax.text()),
            'lightDiv': (self.lightDiv.text()),
            'maxRays': (self.numRays.text()),
            'convThres': (self.convThres.text()),
            'convPlot': self.convPlot.isChecked(),
            'wavMin': (self.wavMin.text()),
            'wavMax': (self.wavMax.text()),
            'enclBox': self.enclosingBox.isChecked(),
            'showSim': self.showSim.isChecked(),
            'saveFolder': self.saveFolder,
            'figDPI': (self.figDPI.text()),
            'resultsFileName': self.saveFileNameShow.text(),
            'inputsFileName': self.saveInputsFile.text()
        })
        folderName = QtWidgets.QFileDialog.getExistingDirectory(self, 'OpenFile')
        fileName = folderName + "/" + self.saveInputsFile.text() + '.txt'
        with open(fileName, 'w') as outfile:
            json.dump(data, outfile)
    
    def onLoadInputs(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        fileName = fileName[0]
        with open(fileName) as json_file:
            data = json.load(json_file)
            for p in data['LSC']:
                self.inputShape.setCurrentText(p['shape'])
                self.STLfile = p['STLfile']
                self.STLfileShow.setPlainText(self.STLfile)
                self.dimx.setText(p['dimX'])
                self.dimy.setText(p['dimY'])
                self.dimz.setText(p['dimZ'])
                solarFacesArr = p['PVedgesLRFB']
                self.solarFaceLeft.setChecked(solarFacesArr[0])
                self.solarFaceRight.setChecked(solarFacesArr[1])
                self.solarFaceFront.setChecked(solarFacesArr[2])
                self.solarFaceBack.setChecked(solarFacesArr[3])
                self.bottomMir.setChecked(p['bottomMir'])
                self.bottomScat.setChecked(p['bottomScat'])
                self.thinFilm.setChecked(p['thinFilm'])
                self.lumophore.setCurrentText(p['lumophore'])
                self.lumophoreConc.setText(p['lumophoreConc'])
                try:
                    self.waveguideAbs.setText(p['waveguideAbs'])
                except:
                    print('no waveguide abs')
                try:
                    self.showSim.setText(p['showSim'])
                except:
                    print('no show t/f')
                self.lightPattern.setCurrentText(p['lightPattern'])
                self.lightDimx.setText(p['lightDimX'])
                self.lightDimy.setText(p['lightDimY'])
                self.lightWavMin.setText(p['lightWavMin'])
                self.lightWavMax.setText(p['lightWavMax'])
                self.lightDiv.setText(p['lightDiv'])
                self.numRays.setText(p['maxRays'])
                self.convThres.setText(p['convThres'])
                self.convPlot.setChecked(p['convPlot'])
                self.wavMin.setText(p['wavMin'])
                self.wavMax.setText(p['wavMax'])
                self.enclosingBox.setChecked(p['enclBox'])
                self.saveFolderShow.setPlainText(p['saveFolder'])
                self.saveFolder = p['saveFolder']
                self.figDPI.setText(p['figDPI'])
                self.saveFileNameShow.setText(p['resultsFileName'])
                self.saveFileName = p['resultsFileName']
                self.saveInputsFile.setText(p['inputsFileName'])
        pass
    
    def onFinishInputClicked(self):
        print("LSC Shape: \t\t" + self.inputShape.currentText())
        print("LSC Dimensions:\t\tLength = " + self.dimx.text() + ", Width = " + self.dimy.text() + ", Height = " + self.dimz.text())
        print("Lumophore:\t\t" + self.lumophore.currentText())
        
        print("Light Pattern:\t\t" + self.lightPattern.currentText())
        print("Light Dimensions:\tLength = " + self.lightDimx.text() + ", Width = " + self.lightDimy.text())
        print("Light Wavelengths:\tMin = " + self.lightWavMin.text() + " nm, Max = " + self.lightWavMax.text() + " nm")
        print("Light Divergence:\t" + self.lightDiv.text() + " deg")
        
        print("Num Rays: \t\t" + self.numRays.text())
        print("Wavelength Range:\tMin = " + self.wavMin.text() + " nm, Max = " + self.wavMax.text() + " nm")
        
        dataFile = ''
        if(self.saveFolder != ''):
            self.saveFileName = self.saveFileNameShow.text()
            dataFile = open(self.saveFolder+'/'+self.saveFileName+'.txt','a')

            dataFile.write("LSC Shape\t" + self.inputShape.currentText() + "\n")
            if(self.inputShape.currentText()=='Import Mesh'):
                dataFile.write("LSC STL\t" + self.STLfile + "\n")
            dataFile.write("LSC Length\t" + self.dimx.text() + "\n")
            dataFile.write("LSC Width\t" + self.dimy.text() + "\n")
            dataFile.write("LSC Height\t" + self.dimz.text() + "\n")
            dataFile.write("Lumophore\t" + self.lumophore.currentText() + "\n")
            dataFile.write("Light Pattern\t" + self.lightPattern.currentText() + "\n")
            dataFile.write("Light Length\t" + self.lightDimx.text() + "\n")
            dataFile.write("Light Width\t" + self.lightDimy.text() + "\n")
            dataFile.write("Light Wav Min\t" + self.lightWavMin.text() + "\n")
            dataFile.write("Light Wav Max\t" + self.lightWavMax.text() + "\n")
            dataFile.write("Light Divergence\t" + self.lightDiv.text() + "\n")
            dataFile.write("Num Rays\t" + self.numRays.text() + "\n")
            dataFile.write("Wavelength Range Min\t" + self.wavMin.text() + "\n")
            dataFile.write("Wavelength Range Max\t" + self.wavMax.text() + "\n")
        
        self.entrance_rays, self.exit_rays, self.exit_norms = self.runPVTrace(dataFile)
        if(self.saveFileName != ''):
            dataFile.close()
        # QApplication.quit()
        
    def runPVTrace(self, dataFile):
        print('Input Received')

        def createWorld(dim):
            world = Node(
            name="World",
            geometry = Sphere(
                radius = 1.1*dim,
                material=Material(refractive_index=1.0),
                )   
            )
            
            return world
        
        def createBoxLSC(dimX, dimY, dimZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Box(
                    (dimX, dimY, dimZ),
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs*1.0), 
                            Scatterer(coefficient = wavAbs*0.0)
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        
        def createCylLSC(dimXY, dimZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Cylinder(
                    dimZ, dimXY/2,
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs), 
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        
        def createSphLSC(dimXYZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Sphere(
                    dimXYZ/2,
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs), 
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        
        def createMeshLSC(self, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Mesh(
                    trimesh = trimesh.load(self.STLfile),
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs*1.00), 
                            Scatterer(coefficient = wavAbs*0.00)
                            ]
                    ),
                ),
                parent = world
            )
            # print(LSC.geometry.trimesh.extents)
            LSC.location = [0,0,0]
            return LSC
        
        def addLR305(LSC, LumConc, LumPLQY):
            wavelength_range = (wavMin, wavMax)
            x = np.linspace(wavMin, wavMax, 200)  # wavelength, units: nm
            absorption_spectrum = lumogen_f_red_305.absorption(x)/10*LumConc  # units: cm-1
            emission_spectrum = lumogen_f_red_305.emission(x)/10*LumConc      # units: cm-1
            LSC.geometry.material.components.append(
                Luminophore(
                    coefficient=np.column_stack((x, absorption_spectrum)),
                    emission=np.column_stack((x, emission_spectrum)),
                    quantum_yield=LumPLQY/100,
                    phase_function=isotropic
                    )
                )
            return LSC, x, absorption_spectrum*10/LumConc, emission_spectrum*10/LumConc
        
        def addBottomSurf(LSC, bottomMir, bottomScat):
            if(bottomMir or bottomScat):
                bottomSpacer = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ/100)
                bottomSpacer.name = "bottomSpacer"
                bottomSpacer.location=[0,0,-(LSCdimZ + LSCdimZ/100)/2]
                bottomSpacer.geometry.material.refractive_index = 1.0
                del bottomSpacer.geometry.material.components[0]
                
            class BottomReflector(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if((bottomMir or bottomScat) and np.allclose(normal, [0,0,-1])):
                        return 1.0
                    
                    return super(BottomReflector, self).reflectivity(surface, ray, geometry, container, adjacent)
                
                def reflected_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if(bottomScat and np.allclose(normal, [0,0,-1])):
                        return tuple(lambertian())
                    return super(BottomReflector, self).reflected_direction(surface, ray, geometry, container, adjacent)
                
                def transmitted_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    
                    return super(BottomReflector, self).transmitted_direction(surface, ray, geometry, container, adjacent)
                
            if(bottomMir or bottomScat):
                bottomSpacer.geometry.material.surface = Surface(delegate = BottomReflector())
            
            return LSC
        
        def addSolarCells(LSC, left, right, front, back, allEdges):
            
            
            
            class SolarCellEdges(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    
                    # if(abs(normal[2]- -1)<0.1 and bottom):
                    #     return 1.0
                    
                    # if(allEdges or left or right or front or back == False):
                    #     return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                    
                    # if(abs(normal[0]- -1)<0.1 and left):
                    #     return 0.0
                    # elif(abs(normal[0]- -1)<0.1 and not left):
                    #     return 1.0
                    
                    # if(abs(normal[0]-1)<0.1 and right):
                    #     return 0.0
                    # elif(abs(normal[0]-1)<0.1 and not right):
                    #     return 1.0
                    
                    # if(abs(normal[1]- -1)<0.1 and front):
                    #     return 0.0
                    # elif(abs(normal[1]- -1)<0.1 and not front):
                    #     return 1.0
                    
                    # if(abs(normal[1]-1)<0.1 and back):
                    #     return 0.0
                    # elif(abs(normal[1]-1)<0.1 and not back):
                    #     return 1.0
                    
                    # if(abs(normal[2])<0.2 and allEdges):
                    #     return 0.0
                    
                    
                    if((allEdges or left or right or front or back) == False):
                        return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                    
                    if(abs(normal[0]- -1)<0.1 and left):
                        return 0.0
                    elif(abs(normal[0]- -1)<0.1 and not left):
                        return 1.0
                    
                    if(abs(normal[0]-1)<0.1 and right):
                        return 0.0
                    elif(abs(normal[0]-1)<0.1 and not right):
                        return 1.0
                    
                    if(abs(normal[1]- -1)<0.1 and front):
                        return 0.0
                    elif(abs(normal[1]- -1)<0.1 and not front):
                        return 1.0
                    
                    if(abs(normal[1]-1)<0.1 and back):
                        return 0.0
                    elif(abs(normal[1]-1)<0.1 and not back):
                        return 1.0
                    
                    if(abs(normal[2])<0.2 and allEdges):
                        return 0.0
                    
                    return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                
                def transmitted_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if(abs(normal[0]- -1)<0.1 and left):
                        return ray.position
                    if(abs(normal[0]-1)<0.1 and right):
                        return ray.position
                    if(abs(normal[1]- -1)<0.1 and front):
                        return ray.position
                    if(abs(normal[1]-1)<0.1 and back):
                        return ray.position
                    if(abs(normal[2])<0.2 and allEdges):
                        return ray.position
                    return super(SolarCellEdges, self).transmitted_direction(surface, ray, geometry, container, adjacent)
            
            LSC.geometry.material.surface = Surface(delegate = SolarCellEdges())
            
            return LSC
        
        def initLight(lightWavMin, lightWavMax):
            h = 6.626e-34
            c = 3.0e+8
            k = 1.38e-23
            
            def planck(wav, T):
                a = 2.0*h*c**2
                b = h*c/(wav*k*T)
                intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
                return intensity
            
            # generate x-axis in increments from 1nm to 3 micrometer in 1 nm increments
            # starting at 1 nm to avoid wav = 0, which would result in division by zero.
            wavelengths = np.arange(lightWavMin*1e-9, lightWavMax*1e-9, 1e-9)
            intensity5800 = planck(wavelengths, 5800.)
            
            dist = Distribution(wavelengths*1e9, intensity5800)
            
            light = Node(
                name = "Light",
                light = Light(
                    wavelength = lambda: dist.sample(np.random.uniform())
                ),
                parent = world
            )
            if(maxZ < 1):
                light.location = (0,0,maxZ*1.1)
            else:
                light.location = (0,0,maxZ/2+0.5)
            light.rotate(np.radians(180), (1, 0, 0))
            return wavelengths*1e9, intensity5800, light
        
        def addRectMask(light, lightDimX, lightDimY):
            light.light.position = functools.partial(rectangular_mask, lightDimX/2, lightDimY/2)
            return light
        
        def addCircMask(light, lightDimX):
            light.light.position = functools.partial(circular_mask, lightDimX/2)
            return light
        
        def addPointSource(light):
            return light
        
        def addLightDiv(light, lightDiv):
            light.light.direction = functools.partial(cone, np.radians(lightDiv))
            return light
            
        def doRayTracing(numRays, convThres, showSim):
            entrance_rays = []
            exit_rays = []
            exit_norms = []
            max_rays = numRays
                
            vis = MeshcatRenderer(open_browser=showSim, transparency=False, opacity=0.5, wireframe=True)
            scene = Scene(world)
            vis.render(scene)
            
            np.random.seed(3)
            
            f = 0
            widgets = [progressbar.Percentage(), progressbar.Bar()]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rays).start()
            history_args = {
                "bauble_radius": LSCdimZ*0.05,
                "world_segment": "short",
                "short_length": LSCdimZ * 0.1,
                }
            k = 0
            if(convPlot):
                fig = plt.figure(num = 4, clear = True)
            xdata = []
            self.ydata = []
            ydataav = 0
            ydataavarr = []
            conv = 1
            self.convarr = []
            edge_emit = 0
            while k < max_rays:
            # while k < 1:
                for ray in scene.emit(1):
                # for ray in scene.emit(int(max_rays)):
                    steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
                    path,surfnorms,events = zip(*steps)
                    if(len(path)<=2):
                        continue
                    if(self.enclosingBox.isChecked() and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT):
                        continue
                    # vis.add_ray_path(path)
                    vis.add_history(steps, **history_args)
                    entrance_rays.append(path[0])
                    if events[-1] in (photon_tracer.Event.ABSORB, photon_tracer.Event.KILL):
                        exit_norms.append(surfnorms[-1])
                        exit_rays.append(path[-1])  
                    elif events[-1] == photon_tracer.Event.EXIT:
                        exit_norms.append(surfnorms[-2])
                        j = surfnorms[-2]
                        if abs(j[2]) <= 0.5:
                            edge_emit+=1
                        exit_rays.append(path[-2]) 
                    f += 1
                    bar.update(f)
                    k+=1
                    xdata.append(k)
                    self.ydata.append(edge_emit/k)
                    ydataav = ydataav*.95 + edge_emit/k * .05
                    ydataavarr.append(ydataav)
                    conv = conv*.95 + abs(edge_emit/k - ydataav)*.05
                    self.convarr.append(conv)
                    if(convPlot):
                        fig = plt.figure(num = 4)
                        if(len(xdata)>2):
                            del xdata[0]
                            del self.ydata[0]
                            del ydataavarr[0]
                            del self.convarr[0]
                        plot(xdata, self.ydata, c='b')
                        plot(xdata, ydataavarr, c='r')
                        plt.grid(True)
                        plt.xlabel('num rays')
                        plt.ylabel('opt. eff')
                        plt.title('optical efficiency vs. rays generated')
                        plt.pause(0.00001)
                        
                        fig = plt.figure(num = 5)
                        plot(xdata, self.convarr, c = 'k')
                        plt.yscale('log')
                        plt.title('convergence')
                        plt.pause(0.00001)
                if(conv < convThres):
                    # numRays = k
                    break
            time.sleep(1)
            vis.render(scene)
            
            return entrance_rays, exit_rays, exit_norms, k
        
        def analyzeResults(entrance_rays, exit_rays, exit_norms):
            edge_emit = 0
            edge_emit_left = 0
            edge_emit_right = 0
            edge_emit_front = 0
            edge_emit_back = 0
            edge_emit_bottom = 0
            edge_emit_top = 0
            entrance_wavs = []
            exit_wavs = []
            emit_wavs = []
            
            for k in exit_norms:
                if k[2]!= None:
                    if((self.rotateY or self.rotateX) is False or enclosingBox):
                        if abs(k[2]) <= 0.5:
                            edge_emit+=1
                        if abs(k[0]- -1)<0.1:
                            edge_emit_left+=1
                        if(abs(k[0]-1)<0.1):
                            edge_emit_right+=1
                        if(abs(k[1]- -1)<0.1):
                            edge_emit_front+=1
                        if(abs(k[1]-1)<0.1):
                            edge_emit_back+=1
                        if(abs(k[2] + 1) < 0.1):
                            edge_emit_bottom+=1
                        if(abs(k[2] - 1) < 0.1):
                            edge_emit_top +=1
                    elif self.rotateX is True:
                        if abs(k[1]) <= 0.5:
                            edge_emit+=1
                        if abs(k[0]- -1)<0.1:
                            edge_emit_left+=1
                        if(abs(k[0]-1)<0.1):
                            edge_emit_right+=1
                        if(abs(k[2]- -1)<0.1):
                            edge_emit_front+=1
                        if(abs(k[2]-1)<0.1):
                            edge_emit_back+=1
                        if(abs(k[1] + 1) < 0.1):
                            edge_emit_bottom+=1
                        if(abs(k[1] - 1)<0.1):
                            edge_emit_top +=1
                    elif self.rotateY is True:
                        if abs(k[0]) <= 0.5:
                            edge_emit+=1
                        if abs(k[2]- -1)<0.1:
                            edge_emit_left+=1
                        if(abs(k[2]-1)<0.1):
                            edge_emit_right+=1
                        if(abs(k[1]- -1)<0.1):
                            edge_emit_front+=1
                        if(abs(k[1]-1)<0.1):
                            edge_emit_back+=1
                        if(abs(k[0] + 1) < 0.1):
                            edge_emit_bottom+=1
                    
            print("\n Optical efficiency: " + str(edge_emit/numRays) + "\n")
            print("\t\tLeft \tRight \tFront \tBack \n")
            print("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
            print("Bottom emission\t" + str(edge_emit_bottom/numRays) + "\t Absorption coeff " + str(-np.log10(edge_emit_bottom/numRays)/float(self.dimz.text())) + "\n")
            print("Top emission\t" + str(edge_emit_top/numRays) +"\n")
            if(self.saveFileName != ''):
                dataFile.write("Opt eff\t" + str(edge_emit/numRays) + "\n")
                dataFile.write("\t\tLeft \tRight \tFront \tBack \n")
                dataFile.write("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
                dataFile.write("type\tposx\tposy\tposz\tdirx\tdiry\tdirz\tsurfx\tsurfy\tsurfz\twav\n")
                dataFile.write("Bottom emission\t" + str(edge_emit_bottom/numRays) + "\t Absorption coeff " + str(-np.log10(edge_emit_bottom/numRays)/float(self.dimz.text())) + "\n")
                dataFile.write("Top emission\t" + str(edge_emit_top/numRays) +"\n")
                for ray in entrance_rays:
                    dataFile.write("entrance\t")
                    for k in range(3):
                        dataFile.write(str(ray.position[k])+"\t")
                    for k in range(3):
                        dataFile.write(str(ray.direction[k])+"\t")
                    for k in range(3):
                        dataFile.write('None \t')
                    dataFile.write(str(ray.wavelength) + "\n")
                for index, ray in enumerate(exit_rays):
                    dataFile.write("exit \t")
                    for k in range(3):
                        dataFile.write(str(ray.position[k])+"\t")
                    for k in range(3):
                        dataFile.write(str(ray.direction[k])+"\t")
                    for k in range(3):
                        dataFile.write(str(exit_norms[index][k])+"\t")
                    dataFile.write(str(ray.wavelength) + "\n")
            xpos_ent = []
            ypos_ent = []
            xpos_exit = []
            ypos_exit = []
            for ray in entrance_rays:
                entrance_wavs.append(ray.wavelength)
                xpos_ent.append(ray.position[0])
                ypos_ent.append(ray.position[1])
            for ray in exit_rays:
                exit_wavs.append(ray.wavelength)
                xpos_exit.append(ray.position[0])
                ypos_exit.append(ray.position[1])
            for k in range(len(exit_wavs)):
                if(exit_wavs[k]!=entrance_wavs[k]):
                    emit_wavs.append(exit_wavs[k])
                    
            
            plt.figure(1, clear = True)
            norm = plt.Normalize(*(wavMin,wavMax))
            wl = np.arange(wavMin, wavMax+1,2)
            colorlist = list(zip(norm(wl), [np.array(wavelength_to_rgb(w))/255 for w in wl]))
            spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)
            colors_ent = [spectralmap(norm(value)) for value in entrance_wavs]
            colors_exit = [spectralmap(norm(value)) for value in exit_wavs]
            scatter(xpos_ent, ypos_ent, alpha=1.0, color=colors_ent)
            scatter(xpos_exit, ypos_exit, alpha=1.0, color=colors_exit)
            plt.title('entrance/exit positions')
            plt.xlabel('x position')
            plt.ylabel('y position')
            plt.axis('equal')
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"xy_plot.png", dpi=figDPI)
            plt.title('Entrance/exit ray positions')
            plt.pause(0.00001)
            
            plt.figure(2, clear = True)
            n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
            plot(wavelengths, intensity/max(intensity)*max(n))
            plt.title('Entrance wavelengths')
            plt.legend()
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"entrance_wavs.png", dpi=figDPI)
            plt.pause(0.00001)
                    
            plt.figure(3, clear=True)
            n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
            if(self.lumophore.currentText() != 'None' ):
                plot(x, abs_spec*max(n), label = 'LR305 abs')
                plot(x, ems_spec*max(n), label = 'LR305 emis')
            plt.title('Re-emitted light wavelengths')
            plt.legend()
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"emit_wavs.png", dpi=figDPI)
            plt.pause(0.00001)
            
            # if(convPlot):
            plt.figure(4)
            if(not convPlot):
                plot(range(len(entrance_rays)), self.ydata)
            plt.title('optical efficiency vs. rays generated')
            plt.grid(True)
            plt.xlabel('num rays')
            plt.ylabel('opt. eff')
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"conv_plot.png", dpi=figDPI)
            plt.pause(0.00001)
            
            plt.figure(5)
            if(not convPlot):
                plot(range(len(entrance_rays)), self.convarr)
            plt.title('convergence')
            plt.grid(True)
            plt.xlabel('num rays')
            plt.ylabel('convergence parameter')
            plt.yscale('log')
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"conv_plot2.png", dpi=figDPI)
            plt.pause(0.00001)

            fig = plt.figure(6, clear=True, figsize=(3, 10))
            fig.add_subplot(515)
            norm = plt.Normalize(*(wavMin,wavMax))
            wl = np.arange(wavMin, wavMax+1,2)
            colorlist = list(zip(norm(wl), [np.array(wavelength_to_rgb(w))/255 for w in wl]))
            spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)
            colors_ent = [spectralmap(norm(value)) for value in entrance_wavs]
            colors_exit = [spectralmap(norm(value)) for value in exit_wavs]
            scatter(xpos_ent, ypos_ent, alpha=1.0, color=colors_ent)
            scatter(xpos_exit, ypos_exit, alpha=1.0, color=colors_exit)
            # plt.title('entrance/exit positions')
            plt.xlabel('x position')
            plt.ylabel('y position')
            plt.axis('equal')
            plt.ylim(-2, 2)
            # plt.title('Entrance/exit ray positions')
            plt.tight_layout()

            fig.add_subplot(513)
            n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
            plot(wavelengths, intensity/max(intensity)*max(n))
            # plt.title('Entrance wavelengths')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('counts (entrance)')
            plt.legend()
            plt.grid()
            plt.pause(0.00001)
            plt.tight_layout()
            
            fig.add_subplot(514)
            n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
            if(self.lumophore.currentText() != 'None' ):
                plot(x, abs_spec*max(n), label = 'LR305 abs')
                plot(x, ems_spec*max(n), label = 'LR305 emis')
            # plt.title('Re-emitted light wavelengths')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('counts (re-emitted)')
            plt.legend(loc='upper left', fontsize='small')
            plt.grid()
            plt.pause(0.00001)
            plt.tight_layout()
            
            fig.add_subplot(511)
            if(not convPlot):
                plot(range(len(entrance_rays)), self.ydata)
            # plt.title('optical efficiency vs. rays generated')
            plt.grid(True)
            plt.xlabel('num rays')
            plt.ylabel('opt. eff.')
            plt.pause(0.00001)
            plt.tight_layout()
            
            fig.add_subplot(512)
            if(not convPlot):
                plot(range(len(entrance_rays)), self.convarr)
            # plt.title('convergence')
            plt.grid(True)
            plt.xlabel('num rays')
            plt.ylabel('convergence')
            plt.yscale('log')
            plt.pause(0.00001)
            plt.tight_layout()

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"plots.png", dpi=figDPI)

            

        #%% define inputs
        wavMin = float(self.wavMin.text())
        wavMax = float(self.wavMax.text())
        LSCdimX = float(self.dimx.text())
        LSCdimY = float(self.dimy.text())
        LSCdimZ = float(self.dimz.text())
        LSCshape = self.inputShape.currentText()
        thinFilm = self.thinFilm.isChecked()
        thinFilmThick = float(self.thinFilmThickness.text())
        LumType = self.lumophore.currentText()
        LumConc = float(self.lumophoreConc.text())
        LumPLQY = float(self.lumophorePLQY.text())
        wavAbs = float(self.waveguideAbs.text())
        wavN = float(self.waveguideN.text())
        lightWavMin = float(self.lightWavMin.text())
        lightWavMax = float(self.lightWavMax.text())
        lightPattern = self.lightPattern.currentText()
        lightDimX = float(self.lightDimx.text())
        lightDimY = float(self.lightDimy.text())
        lightDiv = float(self.lightDiv.text())
        numRays = float(self.numRays.text())
        figDPI = int(self.figDPI.text())
        solAll = self.solarFaceAll.isChecked()
        solLeft = self.solarFaceLeft.isChecked()
        solRight = self.solarFaceRight.isChecked()
        solFront = self.solarFaceFront.isChecked()
        solBack = self.solarFaceBack.isChecked()
        bottomMir = self.bottomMir.isChecked()
        bottomScat = self.bottomScat.isChecked()
        enclosingBox = self.enclosingBox.isChecked()
        LSCbounds = self.LSCbounds
        convPlot = self.convPlot.isChecked()
        convThres = float(self.convThres.text())
        showSim = self.showSim.isChecked()
        
        maxZ = LSCdimZ
        if(LSCshape=='Sphere'):
            maxZ = LSCdimX
        
        world = createWorld(max(LSCdimX, LSCdimY, maxZ))
        
        if(enclosingBox):
            enclBox = createBoxLSC(LSCdimX*1.32, LSCdimY*1.32, LSCdimZ*1.1,0,wavN)
            if(len(widget.LSCbounds)>0):
                
                enclBox.location = [ (self.LSCbounds[0][0] + LSCbounds[1][0])/2, (LSCbounds[0][1] + LSCbounds[1][1])/2, 0]
            enclBox.name = "enclBox"
            enclBox.geometry.material.refractive_index=1.0
            del enclBox.geometry.material.components[0:2]
            enclBox.geometry.material.surface = Surface(delegate = NullSurfaceDelegate())
        
        if not thinFilm:
            if(LSCshape == 'Box'):
                LSC = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Cylinder'):
                LSC = createCylLSC(LSCdimX, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Sphere'):
                LSC = createSphLSC(LSCdimX, wavAbs, wavN)
            if(LSCshape == 'Import Mesh'):
                LSC = createMeshLSC(self, wavAbs, wavN)
                # if(not np.isclose(LSC.location, LSC.geometry.trimesh.centroid).all()):
                #     LSC.translate(-LSC.geometry.trimesh.centroid)
                # LSCmeshdims = LSC.geometry.trimesh.extents
                if(self.rotateY):
                    LSC.rotate(np.radians(90),(0,1,0))
                if(self.rotateX):
                    LSC.rotate(np.radians(90),(1,0,0))
                # if(LSCmeshdims[0] < LSCmeshdims[2]):
                #     LSC.rotate(np.radians(90),(0,1,0))
                #     temp = LSCdimZ
                #     LSCdimZ = LSCdimX
                #     LSCdimX = temp
                #     lightDimX = LSCdimX
                #     maxZ = LSCdimZ
                # elif(LSCmeshdims[1] < LSCmeshdims[2]):
                #     LSC.rotate(np.radians(90),(1,0,0))
                #     temp = LSCdimZ
                #     LSCdimZ = LSCdimY
                #     LSCdimY = temp
                #     lightDimY = LSCdimY
                #     maxZ = LSCdimZ
        else:
            if(LSCshape == 'Box'):
                LSC = createBoxLSC(LSCdimX, LSCdimY, thinFilmThick, wavAbs, wavN)
                bulk_undoped = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Cylinder'):
                LSC = createCylLSC(LSCdimX, thinFilmThick, wavAbs, wavN)
                bulk_undoped = createCylLSC(LSCdimX, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Import Mesh'):
                LSC = createMeshLSC(self, wavAbs, wavN)
                LSC.geometry.trimesh.apply_scale(1,1,thinFilmThick/LSCdimZ)
                bulk_undoped = createMeshLSC(self, wavAbs, wavN)
            LSC.location = (0,0,LSCdimZ/2)
            bulk_undoped.name = "bulk"
            
        if(LumType == 'Lumogen Red'):
            LSC, x, abs_spec, ems_spec = addLR305(LSC, LumConc, LumPLQY)
            
        
        LSC = addSolarCells(LSC, solLeft, solRight, solFront, solBack, solAll)
        
        LSC = addBottomSurf(LSC, bottomMir, bottomScat)
        
        wavelengths, intensity, light = initLight(lightWavMin, lightWavMax)
        if(lightPattern == 'Rectangle Mask'):
            light = addRectMask(light, lightDimX, lightDimY)
        if(lightPattern == 'Circle Mask'):
            light = addCircMask(light, lightDimX)
        if(lightPattern == 'Point Source'):
            light = addPointSource(light)
        if(0<lightDiv<=90):
            light = addLightDiv(light, lightDiv)
            
        
        entrance_rays, exit_rays, exit_norms, numRays = doRayTracing(numRays, convThres, showSim)
        analyzeResults(entrance_rays, exit_rays, exit_norms)
        return entrance_rays, exit_rays, exit_norms
        
        
#%% main
if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
    widget = testingQT()
    widget.show()
    app.exec_()
    
    
