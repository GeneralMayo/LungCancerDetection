import os, shutil, random

ACTIVE_DATA_ROOT = 'Data/ActiveData'
INACTIVE_DATA_ROOT = 'Data/PreprocessedData'

PERCENT_TRAIN = .7
BALANCE_CLASSES = True

#make active data folders
shutil.rmtree(ACTIVE_DATA_ROOT)
os.mkdir(ACTIVE_DATA_ROOT)
os.mkdir(os.path.join(ACTIVE_DATA_ROOT,'Train'))
os.mkdir(os.path.join(ACTIVE_DATA_ROOT,'Test'))

ttNames = ['Test','Train']
mbNames = ['Malignant','Benign']
for ttName in ttNames:
    for mbName in mbNames:
        os.mkdir(os.path.join(ACTIVE_DATA_ROOT,ttName,mbName))

MB_ROOTS = [os.path.join(INACTIVE_DATA_ROOT,'Benign'),os.path.join(INACTIVE_DATA_ROOT,'Malignant')]
#move images to test train folders
for root in MB_ROOTS:
    #get annotation folders (each of which contain patches)
    allContents = os.listdir(root)
    annotationFolders = []
    for item in allContents:
        if(os.path.isdir(os.path.join(root,item))):
            annotationFolders.append(item)

    #split train and test annotation folders
    random.shuffle(annotationFolders)
    splitPoint = int(round(len(annotationFolders)*PERCENT_TRAIN))
    TrainAnnotations = annotationFolders[:splitPoint]
    TestAnnotations = annotationFolders[splitPoint:]
    
    #get image class
    imageClass = root.split('/')[-1]

    #copy train files over
    #src /PreprocessedData/ImageClass/AnnotationName/patchX.npy
    #dst /ActiveData/Train/ImageClass/AnnotationName_FindingX_PatchX.npy
    for annotationFolder in TrainAnnotations:
        folderPath = os.path.join(root, annotationFolder)
        annotationFiles = os.listdir(folderPath)
        for fileName in annotationFiles:
            src = os.path.join(folderPath, fileName)
            dst = os.path.join(ACTIVE_DATA_ROOT,'Train',imageClass,annotationFolder+fileName)
            shutil.copyfile(src,dst)

    #copy test files over
    #src /PreprocessedData/ImageClass/AnnotationName/patchX.npy
    #dst /ActiveData/Test/ImageClass/AnnotationName_FindingX_PatchX.npy
    for annotationFolder in TestAnnotations:
        folderPath = os.path.join(root, annotationFolder)
        annotationFiles = os.listdir(folderPath)
        for fileName in annotationFiles:
            src = os.path.join(folderPath, fileName)
            dst = os.path.join(ACTIVE_DATA_ROOT,'Test',imageClass,annotationFolder+fileName)
            shutil.copyfile(src,dst)


"""          
#balance classes
if(BALANCE_CLASSES):
    ttNames = ['Train','Test']
    for ttName in ttNames:
        minImages = float('inf')
        for root, dirs, files in os.walk(os.path.join(ACTIVE_DATA_ROOT,ttName)):
            if(len(files)>1):
                minImages = min(minImages,len(files))

        for root, dirs, files in os.walk(os.path.join(ACTIVE_DATA_ROOT,ttName)):
            if(len(files)>1):
                numImagesToRemove = int(len(files) - minImages)
                random.shuffle(files)
                for i in range(numImagesToRemove):
                    os.remove(os.path.join(root,files[i]))
"""