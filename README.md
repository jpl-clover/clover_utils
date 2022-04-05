# CLOVER utils

This repo serves 2 key purposes:

1. It contains utility functions and classes that are relevant to the organisation. It serves these utilities as a python package, which is pip installable.
2. It contains the entrypoint functions for our trained models in the ```hubconf.py``` file, thus allowing models to be 'pulled' using PyTorch Hub.

**This repository is public.**

## Usage

Note that you do not need to install this package if you are just seeking to download and use models with PyTorch Hub. That can be accomplished by using the procedure demonstrated in ```tests/test_model_download_and_use.py```. 

If you need to use the modules in this package, then install with pip. In root dir:

```
pip install .
```

If actively editing this package, you may find it easier to instead use:

```
pip install -e .
```

Which will specify that you want to install in editable mode (when you edit the files in the clover_utils package, you do not have to reinstall the package to see the changes).

This is recommended in place of the default ```python setup.py install``` which uses easy_install. If you have an existing install, and want to ensure package and dependencies are updated use ```--upgrade```:

```
pip install --upgrade .
```

To uninstall (use package name):

```
pip uninstall clover_utils
```

## Adding to the utils package

Place your python modules in ```clover_utils/``` alongside the existing python modules. If your modules add additional dependencies, be sure to add them in the ```setup.py``` file in the ```install_requires``` field. Test files including example usages are appreciated.

## Adding new models for public access via PyTorch Hub

1. Host the model weights file in a publically accessible location. For example, in Google Drive, with the sharing permissions set to 'anyone with this link can **view** the file'. This folder has such permissions: https://drive.google.com/drive/u/2/folders/16DP7P8lA0R_T0XY5JrSAMdm--doRJWwr.
2. Identify the url to this file (not folder). If using Google Drive, direct urls can be generated using this tool: https://sites.google.com/site/gdocs2direct/.
3. Create an entrypoint in the ```hubconf.py``` file. An example is provided (```finetuned_supervised_resnet```). Push your changes to this repository.
4. Confirm that the entrypoint is accessible by running ```tests/test_accessible_entrypoints.py```. The entrypoint should be printed to terminal, and can be used as demonstrated in ```tests/test_model_download_and_use.py```.

## Uploading clover_utils package to PyPI

PyPI has certain required meta-data that the ```setup.py``` should provide. To quickly check if your project has this data use:

```
python setup.py check
```

If nothing is reported your package is acceptable.

Create a source distribution. From your root directory:

```
python setup.py sdist
```

This creates a dist/ directory containing a compressed archive of the package (e.g. <PACKAGE_NAME>-<VERSION>.tar.gz in Linux). This file is your source distribution. If it does not automatically contain what you want, then you might consider using a MANIFEST file (see https://docs.python.org/distutils/sourcedist).

Install twine:

```
pip install twine
```

Use twine to upload the created source distrbution on PyPI. You'll need a PyPI account.

```
twine upload dist/*
```
