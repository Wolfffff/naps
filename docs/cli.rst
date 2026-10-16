.. _cli:

Command line interfaces
--------------------------

.. _cli-naps-track:

.. attention::

   Many of the parameters of the command line interfaces are simply passed to OpenCV's ArUco library. For more information on these parameters, please refer to the `ArUco documentation <https://docs.opencv.org/4.7.0/d9/d6a/group__aruco.html>`_.

.. argparse::
   :module: naps.naps_track
   :func: build_parser
   :prog: naps-track

.. _cli-naps-plot:

naps-plot
^^^^^^^^^

.. argparse::
   :module: naps.naps_plot
   :func: build_parser
   :prog: naps-plot

.. _cli-naps-interactions:

naps-interactions
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: naps.naps_interactions
   :func: build_parser
   :prog: naps-interactions
