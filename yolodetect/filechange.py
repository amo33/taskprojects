import os 
import glob 
file1 = glob.glob('*.xml.txt')
file2 = glob.glob("*.xml.txt")

for name in file1:
    if not os.path.isdir(name):
        src = os.path.splitext(name)
        os.rename(name,src[0]+'.txt')
        print(src[0])
for name in file2:
    if not os.path.isdir(name):
        src = os.path.splitext(name)
        os.rename(name,src[0]+'.txt')