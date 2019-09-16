import os

def exist(path,name):
    for root, dirs, files in os.walk(path):
        if name in files:
            return True
        if name in dirs:
            return True
    return False

def find(name,path):
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

def findAllIn(path):
    for root,dirs,files in os.walk(path):
        return files

def ext(path):
    return path.split('.')[-1]

def separe_filename_and_ext(file):
    ## Will fail if there is a '.' in the filename/pathname
    file_sep= file.split('.')
    ext = file_sep[len(file_sep)-1]
    filename = file_sep[:len(file_sep)-1][0]
    return(filename,ext)
