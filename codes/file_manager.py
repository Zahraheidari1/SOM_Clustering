from os import mkdir,listdir
from shutil import copyfile,rmtree
def createdir(dir,l):
    s=set(l)
    for i in s:
        path=f'{dir}\c{i}'
        mkdir(path)
def copy(dir,start,dest):
    for i,j in enumerate(start):
        s=f'{dir}\{j}.txt'
        d=f'{dir}\c{dest[i]}\{j}.txt'
        copyfile(s,d)
def delete_directories(path):
    for f in listdir(path):
        if not '.' in f:
            rmtree(path+f'\{f}')