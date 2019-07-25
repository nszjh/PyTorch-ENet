import os
import cv2
import re
import sys

# Run only if this module is being run directly
if __name__ == '__main__':
    photoFormat = re.compile(r'\w+.(xml)')
    if len(sys.argv) == 2:
        for root, dirs, files in os.walk(sys.argv[1], topdown=False):
            for name in files:
                filePath = os.path.join(root, name)
                match = photoFormat.findall(filePath)
                if len(match):
                    # storage = cv2.FileStorage(filePath, cv2.FILE_STORAGE_READ)
                    #  = storage.getNode("ix1").real()
                    print (filePath)
                    storage = cv2.FileStorage(filePath, cv2.FILE_STORAGE_READ)
                    dataImg = storage.getNode(os.path.basename(filePath).split('.')[0])
                    print (dataImg.mat())
                    storage.release()

    else:
        print ("usage: ConvertAction3D.py  /home")

