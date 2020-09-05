# pvtrace-sv

Modifications made to Daniel Farrell's pvTrace (https://github.com/danieljfarrell/pvtrace) to allow analysis of unconventional LSC geometries.

Added surface normal recording in photon-tracer.

To use, copy photon_tracer.py into the algorithm folder of exiting pvTrace install.

LSC_script.py and LSC_script_parallel_comp.py require manual input of desired input parameters in the code. LSC_script_parallel_comp uses multiprocessing to access all computing cores in a machine. Requires install of multiprocessing.

GUI folder contains main.py that can be run to call a GUI (form.ui) for easier input of simulation parameters. Requires install of PySide2. Tested to work on Anaconda.
