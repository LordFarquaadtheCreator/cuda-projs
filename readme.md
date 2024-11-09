# Compulation Hints
- download nvidia cuda samples directory in user root
- `nvcc -O2 filename.cu -o out-path -l ~/cuda-samples/Common'/`
- [cuda samples repo](https://github.com/NVIDIA/cuda-samples.git)

# Setting up file structure for compilation:
1. Download [cuda-samples library](https://github.com/NVIDIA/cuda-samples.git)
2. Download [thrust library](https://github.com/NVIDIA/thrust.git)
3. Download [CUB library](https://github.com/NVIDIA/cub.git)
4. Copy the actual cub directory IN the repository into your thrust local repository. Thrust needs access specifically to cub/details, it gets confused looking at cub/cub/details.
5. Run ```nvcc -O2 pointIndexing.cu -o pointindexing -I /home/.../cuda-samples/Common/ -I /home/.../thrust/ -I /home/.../cub/```. I don't think I must state the obvious about the directories in this command. 
