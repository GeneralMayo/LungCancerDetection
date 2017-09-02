import os
import csv
import numpy as np
import shutil

import SimpleITK as sitk
from PIL import Image



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



malignantScanInfo = {}
malignantCount = 0
sickPatients = set()
#load malignant annotation info into dict
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

#load benign annotations
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


#make subset dict
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

notFoundCount = 0
allScanInfo = [benignScanInfo,malignantScanInfo]
#loop through malignant an bengin annotations
for mbIndex in range(len(allScanInfo)):
    for annotation in allScanInfo[mbIndex].keys():
        #find patient image path
        annotationPath = ''
        for subsetName in subsetDict.keys():
            if(annotation+'.mhd' in subsetDict[subsetName]):
                annotationPath = os.path.join('Data',subsetName,annotation+'.mhd')
                break

        if(annotationPath == ''):
            print('Not Found')
            notFoundCount+=1
            continue

        #load ct image
        itkImage = sitk.ReadImage(annotationPath)
        origin = itkImage.GetOrigin()
        resolution = itkImage.GetSpacing()

        #loop through annotations in image
        findingNum = 0
        for annotationCoords in allScanInfo[mbIndex][annotation]:
            voxel_coords = [int(np.absolute(annotationCoords[j]-origin[j])/resolution[j]) for j in range(len(annotationCoords))]
            #get pixels
            image = sitk.GetArrayFromImage(itkImage)
            imageDims = image.shape

            #crop
            xStart = voxel_coords[0]-int(IM_SIZE/2)
            xFin = voxel_coords[0]+int(IM_SIZE/2)
            yStart = voxel_coords[1]-int(IM_SIZE/2)
            yFin = voxel_coords[1]+int(IM_SIZE/2)
            if(imageDims[1]<IM_SIZE or imageDims[2]<IM_SIZE):
                raise NameError("ROI larger than image.")
            
            if(xStart<0):
                xStart = 0
                xFin = IM_SIZE
            elif(xFin>imageDims[1]):
                xStart = imageDims[1]-IM_SIZE
                xFin = imageDims[1]
            if(yStart<0):
                yStart = 0
                yFin = IM_SIZE
            elif(yFin>imageDims[2]):
                yStart = imageDims[2]-IM_SIZE
                yFin = imageDims[2]

            
            croppedImage = image[int(voxel_coords[2]), xStart:xFin, yStart:yFin]
            
            #normalize
            maxHU = 400.
            minHU = -1000.
            normalizedImage = (croppedImage - minHU) / (maxHU - minHU)
            normalizedImage[normalizedImage>1] = 1.
            normalizedImage[normalizedImage<0] = 0.

            #save
            if(mbIndex == 0):
                filePath = os.path.join(PREPROCESSED_PATH,'Benign',annotation+'_'+str(findingNum)+'.png')
            else:
                filePath = os.path.join(PREPROCESSED_PATH,'Malignant',annotation+'_'+str(findingNum)+'.png')
            
            print(filePath)
            Image.fromarray(normalizedImage*255).convert('L').save(filePath)

            findingNum+=1

print(notFoundCount)
