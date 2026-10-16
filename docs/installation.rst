.. _installation:

##############################
NAPS Installation Instructions
##############################

.. attention::

    We recommend testing out NAPS using Google Colaboratory before installing NAPS on your local machine. You can find the Google Colaboratory notebook `here <https://colab.research.google.com/github/kocherlab/naps/blob/main/docs/notebooks/naps_basic_workflow.ipynb>`_. Note that this notebook installs a development branch of NAPS (``wolf-dev-fixing-deps``) from GitHub, not the released ``naps-track`` package.

NAPS needs **Python 3.10 or newer**. It reads and writes SLEAP files through `sleap-io <https://github.com/talmolab/sleap-io>`_ and does *not* need the full `SLEAP <https://sleap.ai/>`_ (TensorFlow) installation. You only need SLEAP itself to produce the pose predictions NAPS consumes; install it separately in its own environment.

We recommend installing NAPS in an isolated environment.

***
pip
***

.. code-block:: bash

    pip install naps-track

or, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    uv venv && source .venv/bin/activate
    uv pip install naps-track

.. caution::

    Install exactly one OpenCV distribution. NAPS depends on ``opencv-contrib-python-headless``; a second one (for example ``opencv-python``, pulled in by ``sleap-io[opencv]``) overwrites ``cv2`` and breaks ``cv2.aruco``.

*****
Conda
*****

.. code-block:: bash

    conda create -n naps -c kocherlab -c conda-forge naps-track

.. hint::

    If you haven't installed Conda before, you must install a version of Conda (Anaconda, Miniconda, Miniforge, etc) before running the above snippet. We recommend Miniforge or Miniconda as lightweight versions of Anaconda.

******
GitHub
******

If you are interested in following the development of NAPS, you may want to install the latest development version of NAPS directly from the `GitHub repository <https://github.com/kocherlab/naps>`_.

To do that, do the following:

1. Ensure git is installed:

.. code-block:: bash

    git --version

.. note::

    While many systems have git preinstalled, you may need to install git. This can be done by following the instructions on `git-scm.com/downloads <https://git-scm.com/downloads>`_.

2. Clone the repository:

.. code-block:: bash

    git clone https://github.com/kocherlab/naps.git
    cd naps

3. Install NAPS with the development tools (tests, ruff, ty, pre-commit, build):

.. code-block:: bash

    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev]"
    pytest

Or create the same environment with conda: ``conda env create -f environment.yml``.

***********************
Apple Silicon (macOS)
***********************

The pip and conda instructions above work natively on Apple Silicon; no special environment file is needed any more. Run SLEAP itself following the `SLEAP instructions for Apple Silicon <https://sleap.ai/installation.html#apple-silicon-macs>`_.
