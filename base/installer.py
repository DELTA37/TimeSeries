import requests
import progressbar
import os

def dataset_install(url, path, chunk_size=1000):
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
    if tp == 'application/zip':
        import zipfile
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(os.path.expanduser(path))
        zip_ref.close()
    elif tp == 'application/x-gzip':
        import tarfile
        tar = tarfile.open(filename, 'r')
        tar.extractall(path=path)
        tar.close()

    os.remove(filename)
    print("Done!") 

    
    
