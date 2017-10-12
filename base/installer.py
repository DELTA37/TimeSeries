import requests
import progressbar
import os

def dataset_install(url, path, chunk_size=1000):
    '''
    @param url          : url address of dataset
    @type  url          : str

    @param path         : where locate downloaded dataset
    @type  path         : str

    @param chunk_size   : Optional parameter to download dataset part by part
    @type  chunk_size   : int
    
    @return             : Install dataset into path
    @rtype              : None
    '''
    r = requests.get(url, stream=True)
    chunk_size = int(chunk_size)
    file_size = int(r.headers['Content-Length'])
    tp = r.headers['Content-Type']
    if tp == 'application/zip':
        filename = 'tmp.zip'
    elif tp == 'application/x-gzip':
        filename = 'tmp.tgz'
    
    filename = os.path.expanduser(os.path.join(path, filename))
    f = open(filename, 'wb')
    print("Installing from {} ...".format(url))
    i = 0
    num_bars = int(file_size / chunk_size)
    bar = progressbar.ProgressBar(maxval=num_bars).start()
    for chunk in r.iter_content(chunk_size=chunk_size):
        f.write(chunk)
        bar.update(i)
        i += 1
    f.close()
    print("Extracting ...") 
    unpack(filename, path)
    os.remove(filename)
    print("Done!") 

    
def unpack(path, where_path):
    path = os.path.expanduser(path)
    where_path = os.path.expanduser(where_path)
    ext = os.path.splitext(path)[1]

    if ext == '.zip':
        import zipfile
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(where_path)
        zip_ref.close()

    if is_tarfile(path):
        import tarfile
        tar = tarfile.open(path, 'r')
        tar.extractall(path=where_path)
        tar.close()

    if ext == '.rar':
        import rarfile
        rar = rarfile.RarFile(path)
        rar.extractall(where_path)
        rar.close()


