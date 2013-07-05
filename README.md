AdaBoost
==================

This is a C++ implementation of AdaBoost.

Build
------------------

<h5>Requirement</h5>  
- CMake (http://www.cmake.org/)

1) >cmake .  
2) >make

Data format
------------------

The format of training and test data file is the same as SVM-Light(http://svmlight.joachims.org/) and libsvm(http://www.csie.ntu.edu.tw/~cjlin/libsvm). That is:

    <label> <feature>:<value> <feature>:<value> ... <feature>:<value>  
       .  
       .  
       .  

    <label> = {+1, -1}  
    <feature>: feature index (integer value starting from 1)  
    <value>: feature value (double)'

There are sample data sets at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.

Usage
------------------

<h5>Training</h5>  
    >./abtrain [options] training_set_file [model_file]  
    options:  
      -t: type of boosting (0:discrete, 1:real, 2:gentle) [default:2]  
      -r: the number of rounds [default:100]  
      -v: verbose'

<h5>Prediction</h5>  
    >./abpredict [options] test_set_file model_file  
     options:  
       -o: output score file  
       -v: verbose'
