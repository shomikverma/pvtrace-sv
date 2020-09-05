#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:32:28 2020

@author: shomikverma
"""

import sys
import os

from pvtrace import *
from pvtrace.geometry.utils import EPS_ZERO
from pvtrace.light.utils import wavelength_to_rgb
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
from matplotlib.animation import FuncAnimation
import multiprocessing as mp

def createWorld(l, w, d):
    world = Node(
    name="World",
    geometry = Box(
        (l * 100, w * 100, d * 100),
        material=Material(refractive_index=1.0),
        )   
    )
    
    return world

def createBoxLSC(dimX, dimY, dimZ):
    LSC = Node(
        name = "LSC",
        geometry = 
        Box(
            (dimX, dimY, dimZ),
            material = Material(
                refractive_index = 1.5,
                components = [
                    Absorber(coefficient = 0.525), 
                    ]
            ),
        ),
        parent = world
    )
    
    return LSC

def createCylLSC(dimXY, dimZ):
    LSC = Node(
        name = "LSC",
        geometry = 
        Cylinder(
            dimZ, dimXY/2,
            material = Material(
                refractive_index = 1.5,
                components = [
                    Absorber(coefficient = 0.525), 
                    ]
            ),
        ),
        parent = world
    )
    
    return LSC

def createSphLSC(dimXYZ):
    LSC = Node(
        name = "LSC",
        geometry = 
        Sphere(
            dimXYZ/2,
            material = Material(
                refractive_index = 1.5,
                components = [
                    Absorber(coefficient = 0.525), 
                    ]
            ),
        ),
        parent = world
    )
    
    return LSC

def createMeshLSC():
    LSC = Node(
        name = "LSC",
        geometry = 
        Mesh(
            trimesh = trimesh.load(STLfile),
            material = Material(
                refractive_index = 1.5,
                components = [
                    Absorber(coefficient = 0.525), 
                    ]
            ),
        ),
        parent = world
    )
    # print(LSC.geometry.trimesh.extents)
    return LSC

def addLR305(LSC):
    wavelength_range = (wavMin, wavMax)
    x = np.linspace(wavMin, wavMax, 200)  # wavelength, units: nm
    absorption_spectrum = lumogen_f_red_305.absorption(x)/10*LumConc  # units: cm-1
    emission_spectrum = lumogen_f_red_305.emission(x)/10*LumConc      # units: cm-1
    LSC.geometry.material.components.append(
        Luminophore(
            coefficient=np.column_stack((x, absorption_spectrum)),
            emission=np.column_stack((x, emission_spectrum)),
            quantum_yield=1.0,
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

def addSolarCells(LSC, left, right, front, back, allEdges, bottom):
    
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
            
            
            if(abs(normal[2]- -1)<0.1 and bottom):
                return 1.0
            
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

def planck(wav, T):
    h = 6.626e-34
    c = 3.0e+8
    k = 1.38e-23
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity
dist = None
def wavelengthDist():
    return dist.sample(np.random.uniform())
    
def initLight(lightWavMin, lightWavMax):
    
    # move wavelengthdist and planck back if desired, but gives same results
    
    
    
    # generate x-axis in increments from 1nm to 3 micrometer in 1 nm increments
    # starting at 1 nm to avoid wav = 0, which would result in division by zero.
    wavelengths = np.arange(lightWavMin*1e-9, lightWavMax*1e-9, 1e-9)
    intensity5800 = planck(wavelengths, 5800.)
    global dist
    dist = Distribution(wavelengths*1e9, intensity5800)
    
    
    # lambdadist = lambda: dist.sample(np.random.uniform())
    
    light = Node(
        name = "Light",
        light = Light(
            wavelength = wavelengthDist
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
    
def doRayTracing(numRays):
    entrance_rays = []
    exit_rays = []
    exit_norms = []
    max_rays = numRays
        
    vis = MeshcatRenderer(open_browser=True, transparency=False, opacity=0.5, wireframe=True)
    scene = Scene(world)
    vis.render(scene)
    
    # np.random.seed(3)
    
    # f = 0
    # widgets = [progressbar.Percentage(), progressbar.Bar()]
    # bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rays).start()
    history_args = {
        "bauble_radius": LSCdimZ*0.05,
        "world_segment": "short",
        "short_length": LSCdimZ * 0.1,
        }
    # fig = plt.figure(num = 4, clear = True)
    # ax = plt.axes(xlim =(0, 500),  
    #                 ylim =(0, 1)) 
    # line, = ax.plot([], [], lw=3)
    k = 0
    # if(convPlot):
    #     fig = plt.figure(num = 4, clear = True)
    xdata = []
    ydata = []
    conv = 1
    convarr = []
    edge_emit = 0
    while k < max_rays:
    # while k < 1:
        for ray in scene.emit(1):
        # for ray in scene.emit(max_rays):
            steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
            path,surfnorms,events = zip(*steps)
            if(len(path)<=2):
                continue
            if(enclosingBox and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT):
                continue
            # # vis.add_ray_path(path)
            vis.add_history(steps, **history_args)
            entrance_rays.append(path[0])
            if events[-1] in (photon_tracer.Event.ABSORB, photon_tracer.Event.KILL):
                exit_norms.append(surfnorms[-1])
                exit_rays.append(path[-1])  
            elif events[-1] == photon_tracer.Event.EXIT:
                exit_norms.append(surfnorms[-2])
                # j = surfnorms[-2]
                # if abs(j[2]) <= 0.5:
                #     edge_emit+=1
                exit_rays.append(path[-2]) 
            # f += 1
            # bar.update(f)
            k+=1
            # xdata.append(k)
            # ydata.append(edge_emit/k)
            # if(len(xdata)>2):
            #     del xdata[0]
            #     del ydata[0]
            # if(len(ydata)>2):
            #     conv = conv*.95 + abs(ydata[-1] - ydata[-2])*.05
            # convarr.append(conv)
            # plt.scatter(k, edge_emit/k, c='b')
            # plot(xdata, ydata, c='b')
            # # print(xdata)
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('opt. eff')
            # plt.title('Convergence')
            # plt.pause(0.00001)
                # line.set_data(xdata,ydata)
                # return line,
            # anim = FuncAnimation(fig, animate, interval=20, blit=True)
            
    # time.sleep(1)
    vis.render(scene)
    
    return entrance_rays, exit_rays, exit_norms, ydata, convarr

def RayTracingLoop(ray, scene):
    
    
    # for ray in scene.emit(max_rays):
    steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
    path,surfnorms,events = zip(*steps)
    while(len(path)<=2 or (enclosingBox and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT)):
        for ray in scene.emit(1):
            steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
            path,surfnorms,events = zip(*steps)
    # if(enclosingBox and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT):
    #     continue
    # vis.add_ray_path(path)
    # vis.add_history(steps, **history_args)
    # entrance_rays.append(path[0])
    entrance_rays = path[0]
    if events[-1] in (photon_tracer.Event.ABSORB, photon_tracer.Event.KILL):
        # exit_norms.append(surfnorms[-1])
        exit_norms = surfnorms[-1]
        # exit_rays.append(path[-1])  
        exit_rays = path[-1]
    elif events[-1] == photon_tracer.Event.EXIT:
        # exit_norms.append(surfnorms[-2])
        exit_norms = surfnorms[-2]
        # j = surfnorms[-2]
        # if abs(j[2]) <= 0.5:
        #     # edge_emit+=1
        #     edge_emit = 1
        # exit_rays.append(path[-2]) 
        exit_rays = path[-2]
    else:
        exit_norms = None
        exit_rays = None
    # f += 1
    # bar.update(f)
    # k+=1
    # xdata.append(k)
    # ydata.append(edge_emit/k)
    # if(len(ydata)>2):
    #     conv = conv*.95 + abs(ydata[-1] - ydata[-2])*.05
    # convarr.append(conv)
        
    return entrance_rays, exit_rays, exit_norms
        

def doRayTracingParallel(numRays):
    
    entrance_rays = []
    exit_rays = []
    exit_norms = []
    max_rays = numRays
        
    vis = MeshcatRenderer(open_browser=True, transparency=False, opacity=0.5, wireframe=True)
    scene = Scene(world)
    vis.render(scene)
    
    # np.random.seed(3)
    
    f = 0
    # widgets = [progressbar.Percentage(), progressbar.Bar()]
    # bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rays).start()
    history_args = {
        "bauble_radius": LSCdimZ*0.05,
        "world_segment": "short",
        "short_length": LSCdimZ * 0.1,
        }
    # fig = plt.figure(num = 4, clear = True)
    # ax = plt.axes(xlim =(0, 500),  
    #                 ylim =(0, 1)) 
    # line, = ax.plot([], [], lw=3)
    k = 0
    # if(convPlot):
    #     fig = plt.figure(num = 4, clear = True)
    xdata = []
    ydata = []
    conv = 1
    convarr = []
    edge_emit = 0
    
    pool = mp.Pool(mp.cpu_count())
    
    while(len(set(entrance_rays)) < numRays):
        results = pool.starmap(RayTracingLoop, [(ray, scene) for ray in scene.emit(numRays - len(set(entrance_rays)))])
        
        
        
        # time.sleep(1)
        vis.render(scene)
        
        # print(results)
        global saveData
        saveData = results
        for k in range(len(saveData)):
            entrance_rays.append(saveData[k][0])
            exit_rays.append(saveData[k][1])
            exit_norms.append(saveData[k][2])
    
    
    pool.close()
    unique_entrance = []
    unique_index = []
    for index,ray in enumerate(entrance_rays):
        if ray not in unique_entrance:
            unique_entrance.append(ray)
            unique_index.append(index)
    entrance_rays = unique_entrance
    exit_rays = list(np.array(exit_rays)[unique_index])
    exit_norms = list(np.array(exit_norms)[unique_index])
    # entrance_rays = list(dict.fromkeys(entrance_rays))
    # exit_rays = list(dict.fromkeys(exit_rays))
    # exit_norms = list(dict.fromkeys(exit_norms))
    
    return entrance_rays, exit_rays, exit_norms

def doRayTracingParallel2(numRays):
    entrance_rays = []
    exit_rays = []
    exit_norms = []
    max_rays = numRays
        
    vis = MeshcatRenderer(open_browser=False, transparency=False, opacity=0.5, wireframe=True)
    scene = Scene(world)
    vis.render(scene)
    
    # np.random.seed(3)
    
    # f = 0
    # widgets = [progressbar.Percentage(), progressbar.Bar()]
    # bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rays).start()
    history_args = {
        "bauble_radius": LSCdimZ*0.05,
        "world_segment": "short",
        "short_length": LSCdimZ * 0.1,
        }
    # fig = plt.figure(num = 4, clear = True)
    # ax = plt.axes(xlim =(0, 500),  
    #                 ylim =(0, 1)) 
    # line, = ax.plot([], [], lw=3)
    k = 0
    # if(convPlot):
    #     fig = plt.figure(num = 4, clear = True)
    xdata = []
    ydata = []
    conv = 1
    convarr = []
    edge_emit = 0
    while k < max_rays:
    # while k < 1:
        for ray in scene.emit(1):
        # for ray in scene.emit(max_rays):
            steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
            path,surfnorms,events = zip(*steps)
            # if(len(path)<=2):
            #     continue
            # if(enclosingBox and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT):
            #     continue
            # # vis.add_ray_path(path)
            # vis.add_history(steps, **history_args)
            entrance_rays.append(path[0])
            if events[-1] in (photon_tracer.Event.ABSORB, photon_tracer.Event.KILL):
                exit_norms.append(surfnorms[-1])
                exit_rays.append(path[-1])  
            elif events[-1] == photon_tracer.Event.EXIT:
                exit_norms.append(surfnorms[-2])
                # j = surfnorms[-2]
                # if abs(j[2]) <= 0.5:
                #     edge_emit+=1
                exit_rays.append(path[-2]) 
            # f += 1
            # bar.update(f)
            k+=1
            # xdata.append(k)
            # ydata.append(edge_emit/k)
            # if(len(xdata)>2):
            #     del xdata[0]
            #     del ydata[0]
            # if(len(ydata)>2):
            #     conv = conv*.95 + abs(ydata[-1] - ydata[-2])*.05
            # convarr.append(conv)
            # plt.scatter(k, edge_emit/k, c='b')
            # plot(xdata, ydata, c='b')
            # # print(xdata)
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('opt. eff')
            # plt.title('Convergence')
            # plt.pause(0.00001)
                # line.set_data(xdata,ydata)
                # return line,
            # anim = FuncAnimation(fig, animate, interval=20, blit=True)
            
    # time.sleep(1)
    vis.render(scene)
    
    return entrance_rays, exit_rays, exit_norms, ydata, convarr
def analyzeResults(entrance_rays, exit_rays, exit_norms, ydata, convarr):
    edge_emit = 0
    edge_emit_left = 0
    edge_emit_right = 0
    edge_emit_front = 0
    edge_emit_back = 0
    edge_emit_bottom = 0
    entrance_wavs = []
    exit_wavs = []
    emit_wavs = []
    
    for k in exit_norms:
        if k[2]!= None:
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
            
    print("\n Optical efficiency: " + str(edge_emit/numRays) + "\n")
    print("\t\tLeft \tRight \tFront \tBack \n")
    print("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
    
    if(saveFileName != ''):
        dataFile.write("Opt eff\t" + str(edge_emit/numRays) + "\n")
        dataFile.write("\t\tLeft \tRight \tFront \tBack \n")
        dataFile.write("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
        dataFile.write("type\tposx\tposy\tposz\tdirx\tdiry\tdirz\tsurfx\tsurfy\tsurfz\twav\n")
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
    # colors_ent = entrance_wavs
    # colors_exit = exit_wavs
    scatter(xpos_ent, ypos_ent, alpha=1.0, color=colors_ent)
    scatter(xpos_exit, ypos_exit, alpha=1.0, color=colors_exit)
    plt.title('entrance/exit positions')
    plt.axis('equal')
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"xy_plot.png", dpi=figDPI)
    plt.title('Entrance/exit ray positions')
    plt.show()
    
    plt.figure(2, clear = True)
    n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
    plot(wavelengths, intensity/max(intensity)*max(n))
    plt.title('Entrance wavelengths')
    plt.legend()
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"entrance_wavs.png", dpi=figDPI)
    plt.show()
            
    plt.figure(3, clear=True)
    n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
    plot(x, abs_spec*max(n), label = 'LR305 abs')
    plot(x, ems_spec*max(n), label = 'LR305 emis')
    plt.title('Re-emitted light wavelengths')
    plt.legend()
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"emit_wavs.png", dpi=figDPI)
    plt.show()
    
    # plt.figure(4)
    # plot(range(len(entrance_rays)), ydata)
    # plt.title('optical efficiency vs. rays generated')
    # plt.grid(True)
    # plt.xlabel('num rays')
    # plt.ylabel('opt. eff')
    # if(saveFolder!=''):
    #     plt.savefig(saveFolder+"/"+"conv_plot.png", dpi=figDPI)
    # plt.pause(0.00001)
    
    # plt.figure(5)
    # plot(range(len(entrance_rays)), convarr)
    # plt.title('convergence')
    # plt.grid(True)
    # plt.xlabel('num rays')
    # plt.ylabel('convergence parameter')
    # plt.yscale('log')
    # if(saveFolder!=''):
    #     plt.savefig(saveFolder+"/"+"conv_plot2.png", dpi=figDPI)
    # plt.pause(0.00001)    
    

def analyzeResultsParallel(entrance_rays, exit_rays, exit_norms):
    edge_emit = 0
    edge_emit_left = 0
    edge_emit_right = 0
    edge_emit_front = 0
    edge_emit_back = 0
    edge_emit_bottom = 0
    entrance_wavs = []
    exit_wavs = []
    emit_wavs = []
    
    for k in exit_norms:
        if k[2]!= None:
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
            
    print("\n Optical efficiency: " + str(edge_emit/numRays) + "\n")
    print("\t\tLeft \tRight \tFront \tBack \n")
    print("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
    
    if(saveFileName != ''):
        dataFile.write("Opt eff\t" + str(edge_emit/numRays) + "\n")
        dataFile.write("\t\tLeft \tRight \tFront \tBack \n")
        dataFile.write("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays)+" \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + " \n")
        dataFile.write("type\tposx\tposy\tposz\tdirx\tdiry\tdirz\tsurfx\tsurfy\tsurfz\twav\n")
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
    # colors_ent = entrance_wavs
    # colors_exit = exit_wavs
    scatter(xpos_ent, ypos_ent, alpha=1.0, color=colors_ent)
    scatter(xpos_exit, ypos_exit, alpha=1.0, color=colors_exit)
    plt.title('entrance/exit positions')
    plt.axis('equal')
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"xy_plot.png", dpi=figDPI)
    plt.title('Entrance/exit ray positions')
    plt.show()
    
    plt.figure(2, clear = True)
    n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
    plot(wavelengths, intensity/max(intensity)*max(n))
    plt.title('Entrance wavelengths')
    plt.legend()
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"entrance_wavs.png", dpi=figDPI)
    plt.show()
            
    plt.figure(3, clear=True)
    n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
    plot(x, abs_spec*max(n), label = 'LR305 abs')
    plot(x, ems_spec*max(n), label = 'LR305 emis')
    plt.title('Re-emitted light wavelengths')
    plt.legend()
    if(saveFolder!=''):
        plt.savefig(saveFolder+"/"+"emit_wavs.png", dpi=figDPI)
    plt.show()
    
            
#%% define inputs
    


wavMin = 200
wavMax = 1000
LSCdimX = 6
LSCdimY = 6
LSCdimZ = .32
LSCshape = 'Import Mesh'
STLfolder = '/Users/shomikverma/Documents/Cambridge/CADfiles/final comparison files/'
STLfile = 'LSC_circle_gcode.stl'
STLfilename = STLfile.split('.')[0]
STLfile = STLfolder+STLfile
LumType = 'Lumogen Red'
LumConc = 500
lightWavMin = 300
lightWavMax = 900
lightPattern = 'Rectangle Mask'
lightDimX = LSCdimX+1
lightDimY = LSCdimY+1
lightDiv = 0
numRays = 100
figDPI = 300
solAll = False
solLeft = False
solRight = False
solFront = False
solBack = False
bottomMir = False
saveFolder = ''
rotateY = False
rotateX = False
saveFileName = ''
dataFile = ''
if(saveFolder != ''):
    dataFile = open(saveFolder+'/'+saveFileName+STLfilename+'.txt','a')
enclosingBox = True
saveData = None


maxZ = LSCdimZ
if(LSCshape=='Sphere'):
    maxZ = LSCdimX
    
print('Input Received')

world = createWorld(LSCdimX, LSCdimY, maxZ)


    

if(LSCshape == 'Box'):
    LSC = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ)
if(LSCshape == 'Cylinder'):
    LSC = createCylLSC(LSCdimX, LSCdimZ)
if(LSCshape == 'Sphere'):
    LSC = createSphLSC(LSCdimX)
if(LSCshape == 'Import Mesh'):
    LSC = createMeshLSC()
    LSCmeshdims = LSC.geometry.trimesh.extents
    
    # mesh = LSC.geometry.trimesh
    # LSCdims = mesh.extents
    # LSCbounds = mesh.bounds
    # LSCbounds = LSCbounds - mesh.centroid
    LSCdimX = LSCmeshdims[0]
    LSCdimY = LSCmeshdims[1]
    LSCdimZ = LSCmeshdims[2]
    # time.sleep(1)
    # if(dimx < dimz):
    #     rotateY = True
    #     dimx=LSCdims[2]
    #     dimz = LSCdims[0]
    # elif(dimy < dimz):
    #     rotateX = True
    #     dimy = LSCdims[2]
    #     dimz = LSCdims[1]
    # if(rotateY):
    #     LSC.rotate(np.radians(90),(0,1,0))
    # if(rotateX):
    #     LSC.rotate(np.radians(90),(1,0,0))
        
    if(LSCmeshdims[0] < LSCmeshdims[2]):
        LSC.rotate(np.radians(90),(0,1,0))
        temp = LSCdimZ
        LSCdimZ = LSCdimX
        LSCdimX = temp
        lightDimX = LSCdimX
        maxZ = LSCdimZ
    elif(LSCmeshdims[1] < LSCmeshdims[2]):
        LSC.rotate(np.radians(90),(1,0,0))
        temp = LSCdimZ
        LSCdimZ = LSCdimY
        LSCdimY = temp
        lightDimY = LSCdimY
        maxZ = LSCdimZ
        
if(enclosingBox):
    enclBox = createBoxLSC(LSCdimX*1.01, LSCdimY*1.01, LSCdimZ*1.1)
    enclBox.name = "enclBox"
    enclBox.geometry.material.refractive_index=1.0
    del enclBox.geometry.material.components[0]
    enclBox.geometry.material.surface = Surface(delegate = NullSurfaceDelegate())
    
if(LumType == 'Lumogen Red'):
    LSC, x, abs_spec, ems_spec = addLR305(LSC)

# LSC = addSolarCells(LSC, solLeft, solRight, solFront, solBack, solAll, bottomMir)

wavelengths, intensity, light = initLight(lightWavMin, lightWavMax)
if(lightPattern == 'Rectangle Mask'):
    light = addRectMask(light, lightDimX, lightDimY)
if(lightPattern == 'Circle Mask'):
    light = addCircMask(light, lightDimX)
if(lightPattern == 'Point Source'):
    light = addPointSource(light)
if(0<lightDiv<=90):
    light = addLightDiv(light, lightDiv)


#parallel
compiletime2 = []
start_time2 = time.time()

# entrance_rays, exit_rays, exit_norms = doRayTracingParallel(numRays)

end_time2 = time.time()
print("---  parallel %s seconds ---" % (end_time2 - start_time2))
compiletime2.append(end_time2 - start_time2)

# analyzeResultsParallel(entrance_rays, exit_rays, exit_norms)

# non parallel

compiletime= []
start_time1 = time.time()

entrance_rays, exit_rays, exit_norms, ydata, convarr = doRayTracing(numRays)


end_time1 = time.time()

print("--- non parallel %s seconds ---" % (end_time1 - start_time1))
compiletime.append(end_time1 - start_time1)
analyzeResults(entrance_rays, exit_rays, exit_norms, ydata, convarr)


if(saveFolder != ''):
    dataFile.close()
# return entrance_rays, exit_rays, exit_norms

print("average compile time" + str(np.mean(compiletime)))
print("average compile time" + str(np.mean(compiletime2)))
    
print("--- parallel %s seconds ---" % (end_time2 - start_time2))
print("--- non parallel %s seconds ---" % (end_time1 - start_time1))