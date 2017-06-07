# KNN_experiment
The profile is used to do a simple experiment to compare two KNN algorithms: kd-tree and LSH.
The experiment uses the datasets from http://corpus-texmex.irisa.fr/#matlab.

## prerequisite:

FLANN:http://www.cs.ubc.ca/research/flann/

FALCONN:https://github.com/FALCONN-LIB/FALCONN

C++ on Mac or Unix.(It works on my mac osx).

### How to run the test:

```
g++ -std=c++0x ./KNN_test.cpp -I ./falconn/src/include -I ./falconn/external/eigen -o knntest
```

The line assumes you put the FALCONN and FLANN source direction in the same file with the KNN_test.cpp.

You might add

```
-I ./flann/flann.hpp
```

to include FLANN.

In the file KNN_test.cpp you can tune the some parameter, e.g., NUM_DATA, NUM_HASH_TABLE… to perform various test.