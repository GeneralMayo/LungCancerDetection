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

#move images to test train folders
for root, dirs, files in os.walk(INACTIVE_DATA_ROOT):
    if(len(files)>1):
        random.shuffle(files)

        splitPoint = int(round(len(files)*PERCENT_TRAIN))
        TrainImages = files[:splitPoint]
        TestImages = files[splitPoint:]
        
        #get class mapping
        imageClass = root.split('/')[-1]

        #move data
        for fileName in TrainImages:
            src = root+'/'+fileName
            dst = os.path.join(ACTIVE_DATA_ROOT,'Train',imageClass,fileName)
            shutil.copyfile(src,dst)


        for fileName in TestImages:
            src = root+'/'+fileName
            dst = os.path.join(ACTIVE_DATA_ROOT,'Test',imageClass,fileName)
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