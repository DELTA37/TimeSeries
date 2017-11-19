# TimeSeries

We provide own pytorch wrap

base: 
  - network      # BaseNet : provide get_trainable and get_restorable methods
  - layers       # Layer   : provide get_trainable and get_restorable methods
  - reader       # Reader  : wrap for pytorch dataset, need to use file configuration
  - transform    # provide a numpy and putorch version of non parametric, non trainable transformation
  - writer       # adapter to tensorboard
  - installer    # use it when you want download and parse your specific dataset  

contrib:
  - utils
  - shape_utils
train            # lr save_num num epoch optimizer config  
test             # num config  
config           # all configuration  

You have to build your model inheriting from a BaseNet, after that you must describe your model in __init__ method,   
also you must create your own reader inheriting Reader and create your own Dataset class inheriting from Dataset  
  
Then you can launch training and testing by   
  - python3.6 train.py <parameters>  
  - python3.6 test.py <parameters>  
  
# HOW TO PLOT DATA FROM MOEX

We provide communication with http://www.moex.com with the help of open-api. 
To plot graph of the security you want, you need to install [requirements.txt](https://github.com/Kakoedlinnoeslovo/TimeSeries/blob/master/requirements.txt), to do this: 
```
virtualenv -p python3 envname
pip install --upgrade virtualenv
pip install -r requirements.txt
```
Some example:
```
(timenv) MacBook-Air-Roman:TimeSeries romandegtyarev$ python plotter.py ABRD
```  
will plot something like that:
![Example](/examples/PRICES_ABRD.png?raw=true "Title")


# Our wiki:  
https://github.com/DELTA37/TimeSeries/wiki/Articles  
