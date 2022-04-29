# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:52:02 2021

@author: U062797
"""

import os
from aicspylibczi import CziFile
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import color
from skimage import morphology

# input parameters
finalScale=0.1
n_cols=4
n_rows=5
background=(1.0, 1.0, 1.0)


# file access
path=""
initFolder=""
saveFolder=""

imgPath=os.path.join(path, initFolder)
savePath=os.path.join(path, saveFolder)

finalROIList=[]
for imgFile in os.listdir(imgPath):
    if imgFile.endswith(".czi"):
        bloc=os.path.splitext(imgFile)[0].split('_')[0]
        slide=os.path.splitext(imgFile)[0].split('_')[1]
        
        # Open image
        czi=CziFile(os.path.join(imgPath, imgFile))
        
        # Extract full mosaic image boundaries 
        mosaic_bbox=czi.get_mosaic_bounding_box()
        yOr=mosaic_bbox.y
        xOr=mosaic_bbox.x
        yMax= mosaic_bbox.h
        xMax= mosaic_bbox.w
        
        # Create a list of ROIs (grid) based on n_cols/n_rows
        roiList=[]
        roiPosition=0
        for row in range(n_rows):
            for col in range(n_cols):
                
                y0=yOr+int(row*yMax/n_rows)
                y1=yOr+int((row+1)*yMax/n_rows)
                x0=xOr+int(col*xMax/n_cols)
                x1=xOr+int((col+1)*xMax/n_cols)
                
                roiList.append([roiPosition,[x0,x1,y0,y1]])
                roiPosition+=1
        
        # Read scenes info and attribute each scene to a ROI
        scenes_bbox=czi.get_all_scene_bounding_boxes()
        scenesList=list(scenes_bbox.keys())
        
        sceneDict={"n_scene":[], "n_roi":[]}
        
        for scene in scenesList:
            xCoG=scenes_bbox[scene].x+int(scenes_bbox[scene].w/2)
            yCoG=scenes_bbox[scene].y+int(scenes_bbox[scene].h/2)
            
            for roi in roiList:
                if np.logical_and(
                        np.logical_and(xCoG>=roi[1][0], xCoG<roi[1][1]), 
                        np.logical_and(yCoG>=roi[1][2], yCoG<roi[1][3])
                        ) == True:
                    sceneDict["n_scene"].append(scene)
                    sceneDict["n_roi"].append(roi[0])
                else:
                    pass
        
        # Create individual image and check is some ROIs are empty
        for roi in range(n_cols*n_rows):
            ROI = "ROI%d" %roi
            if roi not in sceneDict["n_roi"]:
                pass
            
            else:
                sceneIdxList=[idx for idx, scene in enumerate(sceneDict["n_roi"]) if scene == roi]
                mosaicCoordDict={"x0":[], "x1":[], "y0":[], "y1":[]}
                
                for idx in sceneIdxList:
                    sceneNo=sceneDict["n_scene"][idx]
                    # For each scene of the ROI, access x,y coordinates and place them in a dict
                    mosaicCoordDict["x0"].append(scenes_bbox[sceneNo].x)
                    mosaicCoordDict["x1"].append((scenes_bbox[sceneNo].w+scenes_bbox[sceneNo].x))
                    mosaicCoordDict["y0"].append(scenes_bbox[sceneNo].y)
                    mosaicCoordDict["y1"].append((scenes_bbox[sceneNo].h+scenes_bbox[sceneNo].y))
                
                # Only keep min and max 
                mosaicCoordDict["x0"]=min(mosaicCoordDict["x0"])
                mosaicCoordDict["x1"]=max(mosaicCoordDict["x1"])
                mosaicCoordDict["y0"]=min(mosaicCoordDict["y0"])
                mosaicCoordDict["y1"]=max(mosaicCoordDict["y1"])
                
                # Create a white image and put scenes into it
                whiteFrame=255*np.ones((int((mosaicCoordDict["y1"]-mosaicCoordDict["y0"])*finalScale), int((mosaicCoordDict["x1"]-mosaicCoordDict["x0"])*finalScale), 3), np.uint8)
                # whiteFrame=np.ones((int((mosaicCoordDict["y1"]-mosaicCoordDict["y0"])*finalScale), int((mosaicCoordDict["x1"]-mosaicCoordDict["x0"])*finalScale)), np.uint16)
               
                for idx in sceneIdxList:
                    sceneNo=sceneDict["n_scene"][idx]
                    imgArray=czi.read_mosaic((scenes_bbox[sceneNo].x, scenes_bbox[sceneNo].y, scenes_bbox[sceneNo].w, scenes_bbox[sceneNo].h), scale_factor=finalScale, C=0, background_color=background)
                    imgArray=np.squeeze(imgArray, axis=0)

                    whiteFrame[int((scenes_bbox[sceneNo].y-mosaicCoordDict["y0"])*finalScale):int((scenes_bbox[sceneNo].y-mosaicCoordDict["y0"])*finalScale+imgArray.shape[0]), 
                                int((scenes_bbox[sceneNo].x-mosaicCoordDict["x0"])*finalScale):int((scenes_bbox[sceneNo].x-mosaicCoordDict["x0"])*finalScale+imgArray.shape[1])
                                ]=imgArray
                    
                    # Convert bgr to rgb to keep the same colors as original image
                    rgb = whiteFrame[..., ::-1]
                    
                # Save final image
                filename=bloc+'_'+ROI+'_'+ slide +".tiff"
                
                io.imsave(os.path.join(savePath, filename), rgb)

            if (bloc+'_'+ROI) not in finalROIList:
                    finalROIList.append(bloc+'_'+ROI)

                
print ("[INFO] All slices are now cut into ROIs")
