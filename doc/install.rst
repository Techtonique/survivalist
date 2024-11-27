Installing survivalist
==========================

The recommended and easiest way to install survivalist is to use
:ref:`install-conda` or :ref:`install-pip`.
Pre-built binary packages for survivalist are available for Linux, macOS, and Windows.
Alternatively, you can install survivalist :ref:`install-from-source`.

.. _install-conda:

conda
-----

If you have `conda <https://docs.anaconda.com/>`_ installed, you can
install survivalist from the ``conda-forge`` channel by running::

  conda install -c conda-forge survivalist

.. _install-pip:

pip
---

If you use ``pip``, install the latest release of survivalist with::

  pip install survivalist


.. _install-from-source:

From Source
-----------

If you want to build survivalist from source, you
will need a C/C++ compiler to compile extensions.

**Linux**

On Linux, you need to install *gcc*, which in most cases is available
via your distribution's packaging system.
Please follow your distribution's instructions on how to install packages.

**macOS**

On macOS, you need to install *clang*, which is available from
the *Command Line Tools* package. Open a terminal and execute::

  xcode-select --install

Alternatively, you can download it from the
`Apple Developers page <https://developer.apple.com/downloads/index.action>`_.
Log in with your Apple ID, then search and download the
*Command Line Tools for Xcode* package.

**Windows**

On Windows, the compiler you need depends on the Python version
you are using. See `this guide <https://wiki.python.org/moin/WindowsCompilers>`_
to determine which Microsoft Visual C++ compiler to use with a specific Python version.


Latest Release
^^^^^^^^^^^^^^

To install the latest release of survivalist from source, run::

  pip install survivalist --no-binary survivalist


.. note::

    If you have not installed the :ref:`dependencies <dependencies>` previously, this command
    will first install all dependencies before installing survivalist.
    Therefore, installation might fail if build requirements of some dependencies
    are not met. In particular, `osqp <https://github.com/oxfordcontrol/osqp-python>`_
    does require `CMake <https://cmake.org/>`_ to be installed.

Development Version
^^^^^^^^^^^^^^^^^^^

To install the latest source from our `GitHub repository <https://github.com/sebp/survivalist/>`_,
you need to have `Git <https://git-scm.com/>`_ installed and
simply run::

  pip install git+https://github.com/sebp/survivalist.git



.. _dependencies:

Dependencies
------------

The current minimum dependencies to run survivalist are:

- Python 3.10 or later
- ecos
- joblib
- numexpr
- numpy
- osqp
- pandas 1.4.0 or later
- scikit-learn 1.4 or 1.5
- scipy
- C/C++ compiler
