# utils
Utility functions and classes for the CLOVER organisation.

## Usage

Note that you do not need to install this package if you are just seeking to download and use models with pytorch hub. That can be accomplished by using the procedure outlined in ```test_model_download.py```. 

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

Place your python modules in ```clover_utils/``` alongside the existing python modules. If your modules add additional dependencies, be sure to add them in the ```setup.py``` file in the ```install_requires``` field. 

## Adding new models for public access via PyTorch Hub

1. Host the model weights in a publically accessible location. For example, in Google Drive, with the permissions set to 'anyone with this link can download the file'.
2. Identify the url to this file. If using Google Drive, direct urls can be generated using this tool: https://sites.google.com/site/gdocs2direct/.
3. Create an entrypoint in the ```hubconf.py``` file. An example is provided (```finetuned_supervised_resnet```). Push your changes to this repository.
4. Confirm that the entrypoint is accessible by running ```test_accessible_entrypoints.py```. The entrypoint should be printed to terminal, and can be used as demonstrated in ```test_model_download_and_use.py```.

## Uploading to PyPI

PyPI has certain required meta-data that the setup.py should provide. To quickly check if your project has this data use:

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