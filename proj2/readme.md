# How To 
- in order to compile the cuda code to a python library you gotta do the following
- make directory `build`
- run `cmake` from `proj2` library in `build`
- this generates a `makefile`
- run `make`
- copy over `haversine_library.so` to `build`
- now python can import the library in `test_3cities.py`
