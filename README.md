# TimeSeries

base: 
  - network
  - layers
  - reader
  - transform
  - writer

train

You have to build your model inheriting from a BaseNet, after that you must describe your model in __init__ method,   
also you must create your own reader and override formalise()

Then you can launch training and testing by 
  - python3.6 train.py <parameters>
  - python3.6 test.py <parameters>
  
We will describe parameters when we finish project
