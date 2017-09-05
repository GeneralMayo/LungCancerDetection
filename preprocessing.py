import os
import csv
import numpy as np
import shutil

import SimpleITK as sitk
from PIL import Image
from  scipy import ndimage


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing 
    voxel_coordinates = voxel_coordinates.astype(np.int)

    return voxel_coordinates

def get_xy_slices(image3d, voxelCenters, patchSize):
    #get xyz shape
    shape = image3d.shape

    layeredPatches = np.zeros((3,patchSize,patchSize))
    for (num,voxelCenter) in enumerate(voxelCenters):
        #get index ranges
        xStart = voxelCenter[2] - patchSize/2
        xFin = voxelCenter[2] + patchSize/2
        yStart = voxelCenter[1] - patchSize/2
        yFin = voxelCenter[1] + patchSize/2
        z = voxelCenter[0]

        #check bounds
        if(xStart<0 or xFin>=shape[2]
            or yStart<0 or yFin>=shape[1]
            or z<0 or z>=shape[0]):
            return []

        #get image slice
        layeredPatches[num,:,:] = image3d[z,yStart:yFin,xStart:xFin]

    return [layeredPatches]

def get_yz_slices(image3d, voxelCenters, patchSize):
    #z,y,x shape
    shape = image3d.shape

    layeredPatches = np.zeros((3,patchSize,patchSize))
    for (num,voxelCenter) in enumerate(voxelCenters):
        #get index ranges
        x = voxelCenter[2]
        yStart = voxelCenter[1] - patchSize/2
        yFin = voxelCenter[1] + patchSize/2
        zStart = voxelCenter[0] - patchSize/2
        zFin = voxelCenter[0] + patchSize/2

        #check bounds
        if(x<0 or x>=shape[2]
            or yStart<0 or yFin>=shape[1]
            or zStart<0 or zFin>=shape[0]):
            return []

        #get image slice
        layeredPatches[num,:,:] = image3d[zStart:zFin,yStart:yFin,x]

    return [layeredPatches]

def get_xz_slices(image3d, voxelCenters, patchSize):
    #get zyx shape
    shape = image3d.shape

    layeredPatches = np.zeros((3,patchSize,patchSize))
    for (num, voxelCenter) in enumerate(voxelCenters):
        #get index ranges
        xStart = voxelCenter[2] - patchSize/2
        xFin = voxelCenter[2] + patchSize/2
        y = voxelCenter[1]
        zStart = voxelCenter[0] - patchSize/2
        zFin = voxelCenter[0] + patchSize/2

        #check bounds
        if(xStart<0 or xFin>=shape[2]
            or y<0 or y>=shape[1]
            or zStart<0 or zFin>=shape[0]):
            return []

        #get image slice
        layeredPatches[num,:,:] = image3d[zStart:zFin,y,xStart:xFin]

    return [layeredPatches]


def get_relevant_image_slices(image3d, annotationWorldCoordinates, origin, spacing, patchSize):
    allPatches = []
    #iterate through planes
    for planeNum in range(3):
        #iterate through translations of -2,0,2
        voxelCenters = []
        for traslation in np.linspace(-3,3,3):
            #get center of patch
            sliceCenterCoordinates = np.copy(annotationWorldCoordinates)
            sliceCenterCoordinates[planeNum] += traslation

            #get voxel center
            voxelCenters.append(world_2_voxel(sliceCenterCoordinates, origin, spacing))

        #get patches
        if(planeNum == 0):
            allPatches += get_yz_slices(image3d, voxelCenters, patchSize)
        elif(planeNum == 1):
            allPatches += get_xz_slices(image3d, voxelCenters, patchSize)
        else:
            allPatches += get_xy_slices(image3d, voxelCenters, patchSize)

    return allPatches

def resample(imageArray, oldSpacing, RESIZE_SPACING=[1,1,1]):
    shape = imageArray.shape

    #calculate resize factor and new shape
    resize_factor = oldSpacing / RESIZE_SPACING
    new_real_shape = oldShape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / oldShape
    new_spacing = oldSpacing / real_resize

    #resize image
    newImageArray = ndimage.interpolation.zoom(imageArray, real_resize)

    return (newImageArray, new_spacing)

def normalize(imageArray):
    #set pixels outside bounds of scanner to 0
    imageArray[imageArray == -2000] = 0

    #normalize image
    maxHU = 400.
    minHU = -1000.
    normalizedImage = (imageArray - minHU) / (maxHU - minHU)
    normalizedImage[normalizedImage>1] = 1.
    normalizedImage[normalizedImage<0] = 0.
    return normalizedImage

def main():
    CSV_PATH = "Data/CSVFILES"
    MALIGNANT_CSV = "annotations.csv"
    ALL_CSV = "candidates.csv"
    PREPROCESSED_PATH = "Data/PreprocessedData"
    IM_SIZE = 50
    ANNOTATIONS_PER_PATIENT = 5


    #make directories
    shutil.rmtree(PREPROCESSED_PATH)
    os.mkdir(PREPROCESSED_PATH)
    os.mkdir(os.path.join(PREPROCESSED_PATH,'Malignant'))
    os.mkdir(os.path.join(PREPROCESSED_PATH,'Benign'))


    #load malignant annotations (keys are individual patients)
    malignantScanInfo = {}
    malignantCount = 0
    sickPatients = set()
    with open(os.path.join(CSV_PATH,MALIGNANT_CSV), 'rb') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        #row example = seriesuid,coordX,coordY,coordZ,diameter_mm
        for row in csvReader:
            if(row[0].split(".")[0] == '1'):
                if(row[0] in sickPatients):
                    malignantScanInfo[row[0]].append((float(row[1]),float(row[2]),float(row[3])))
                else:
                    sickPatients.add(row[0])
                    malignantScanInfo[row[0]] = [(float(row[1]),float(row[2]),float(row[3]))]
                malignantCount+=1

    #load benign annotations (keys are individual patients)
    benignScanInfo = {}
    benignCount = 0
    healthyPatients = set()
    with open(os.path.join(CSV_PATH,ALL_CSV), 'rb') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if(row[0].split(".")[0] == '1' and not(row[0] in sickPatients)):
                if(row[0] in healthyPatients):
                    if(len(benignScanInfo[row[0]])>ANNOTATIONS_PER_PATIENT):
                        continue
                    else:
                        benignScanInfo[row[0]].append((float(row[1]),float(row[2]),float(row[3])))
                else:
                    healthyPatients.add(row[0])
                    benignScanInfo[row[0]] = [(float(row[1]),float(row[2]),float(row[3]))]
                benignCount+=1
                if(benignCount >= malignantCount):
                    break

    #Map data subsets to a sets of patients in those subsets
    subsetDict = {}
    totalFiles = 0
    for setNum in range(10):
        subsetDir = 'subset'+str(setNum)
        for root, dirs, files in os.walk(os.path.join('Data',subsetDir)):
            for fileName in files:
                if(fileName.split('.')[-1]=='mhd'):
                    if(subsetDir in subsetDict):
                        subsetDict[subsetDir].add(fileName)
                    else:
                        subsetDict[subsetDir] = set([fileName])
                    totalFiles +=1

    #preprocess annotations
    notFoundCount = 0
    allScanInfo = [benignScanInfo, malignantScanInfo]
    for mbIndex in range(len(allScanInfo)):
        for annotation in allScanInfo[mbIndex].keys():
            
            #find patient image path
            annotationPath = ''
            for subsetName in subsetDict.keys():
                if(annotation+'.mhd' in subsetDict[subsetName]):
                    annotationPath = os.path.join('Data',subsetName,annotation+'.mhd')
                    break

            #some files didn't download correctly
            if(annotationPath == ''):
                print('Not Found')
                notFoundCount+=1
                continue


            #load ct image and get meta data
            itkImage = sitk.ReadImage(annotationPath)
            imageArray = sitk.GetArrayFromImage(itkImage)
            #change x,y,z to z,y,x
            origin = np.array(list(reversed(itkImage.GetOrigin())))
            #change x,y,z to z,y,x
            resolution = np.array(list(reversed(itkImage.GetSpacing())))

            #loop through annotations in image
            findingNum = 0
            for annotationCoords in allScanInfo[mbIndex][annotation]:
                
                #change x,y,z to z,y,x
                annotationCoords = np.array([annotationCoords[2],annotationCoords[1],annotationCoords[0]])
                
                #get patches
                patches = get_relevant_image_slices(imageArray, annotationCoords, origin, resolution, IM_SIZE)

                """
                for i in range(3):
                    for j in range(3):
                        patchSlice = patches[i][j]

                        patchSlice = normalize(patchSlice)
                        Image.fromarray(patchSlice*255).convert('L').save('Test_'+str(i)+str(j)+'.png')
                input()
                """
                #normalize patches
                for (num,patch) in enumerate(patches):
                    patches[num] = normalize(patch)

                #save patchs
                annotationPath = ''
                if(mbIndex == 0):
                    annotationPath = os.path.join(PREPROCESSED_PATH,'Benign',annotation+'_Finding'+str(findingNum))
                else:
                    annotationPath = os.path.join(PREPROCESSED_PATH,'Malignant',annotation+'_Finding'+str(findingNum))

                os.mkdir(annotationPath)

                for (num , patch) in enumerate(patches):
                    filePath = os.path.join(annotationPath,'Patch'+str(num)+'.npy')
                    np.save(filePath, patch)

                findingNum+=1

    print(notFoundCount)

if __name__ == '__main__':
    main()



