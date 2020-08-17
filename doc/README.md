How to build the docs?
======================

Building the docs requires installing the dependencies listed in `requirements.txt`.

The building process involves conversion of Jupyter notebooks to `rst` files. To facilitate this you need to set `NB_KERNEL` environment variable to the desired python kernel (e.g. `export NB_KERNEL="python3"`). To get a list of python kernels available, run `jupyter kernelspec list`.

After you're set, from the `doc` directory run `make notebooks html` to convert all notebooks, build all gallery examples and the documentation iteslf. The designated output folder is `_build/html`. 

Run `make clean` to delete a built documentation.