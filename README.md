# utils
Utility functions and classes for the CLOVER organisation.

## Usage

Install with pip. In root dir:

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

## Adding to this

Place your python modules in ```clover_utils/``` alongside the existing python modules. If your modules add additional dependencies, be sure to add them in the ```setup.py``` file in the ```install_requires``` field. 

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