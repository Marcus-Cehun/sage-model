#!/usr/bin/env python
"""
Main driver script to handle plotting the output of the ``SAGE`` model.  Multiple models
can be placed onto the plots by extending the variables in the ``__main__`` function call.

To add your own data format, create a subclass module (e.g., ``sage_binary.py``) and add an
option to ``Results.__init__``.  This subclass module needs methods ``set_cosmology()``,
``determine_num_gals()`` and  ``read_gals()``.

To calculate and plot extra properties, first add the name of your new plot to the
``plot_toggles`` dictionary.  You will need to create a method in ``model.py`` to
calculate your properties and name it ``calc_<Name of your plot toggle>``.  To plot your
new property, you will need to create a function in ``plots.py`` called ``plot_<Name of
your plot toggle>``.

For example, to generate and plot data for the ``SMF`` plot, we have methods ``calc_SMF()``
and ``plot_SMF()``.

Refer to the documentation inside the ``model.py`` and ``plot.py`` modules for more
details.

Author: Jacob Seiler
"""

import plots as plots

# Import the subclasses that handle the different SAGE output formats.
from sage_binary import SageBinaryModel

try:
    from sage_hdf5 import SageHdf5Model
except ImportError:
    print("h5py not found.  If you're reading in HDF5 output from SAGE, please install "
          "this package.")

import numpy as np

# Sometimes we divide a galaxy that has zero mass (e.g., no cold gas). Ignore these
# warnings as they spam stdout.
old_error_settings = np.seterr()
np.seterr(all="ignore")

# Refer to the project for a list of TODO's, issues and other notes.
# https://github.com/sage-home/sage-model/projects/7

class Results:
    """
    Defines all the parameters used to plot the models.

    Attributes
    ----------

    num_models : Integer
        Number of models being plotted.

    models : List of ``Model`` class instances with length ``num_models``
        Models that we will be plotting.  Depending upon the format ``SAGE`` output in,
        this will be a ``Model`` subclass with methods to parse the specific data format.

    plot_toggles : Dictionary
        Specifies which plots will be generated. An entry of `1` denotes
        plotting, otherwise it will be skipped.

    plot_output_path : String
        Base path where the plots will be saved.

    plot_output_format : String
        Format the plots are saved as.
    """

    def __init__(self, all_models_dict, plot_toggles, plot_output_path="./plots",
                 plot_output_format=".png", debug=False):
        """
        Initialises the individual ``Model`` class instances and adds them to
        the ``Results`` class instance.

        Parameters
        ----------

        all_models_dict : Dictionary
            Dictionary containing the parameter values for each ``Model``
            instance. Refer to the ``Model`` class for full details on this
            dictionary. Each field of this dictionary must have length equal to
            the number of models we're plotting.

        plot_toggles : Dictionary
            Specifies which plots will be generated. An entry of 1 denotes
            plotting, otherwise it will be skipped.

        plot_output_path : String, default "./plots"
            The path where the plots will be saved.

        plot_output_format : String, default ".png"
            Format the plots will be saved as.

        debug : {0, 1}, default 0
            Flag whether to print out useful debugging information.

        Returns
        -------

        None.
        """
        self.IDs_to_Process = None
        self.num_models = len(all_models_dict["model_path"])
        self.plot_output_path = plot_output_path

        if not os.path.exists(self.plot_output_path):
            os.makedirs(self.plot_output_path)

        self.plot_output_format = plot_output_format

        # We will create a list that holds the Model class for each model.
        all_models = []

        # Now let's go through each model, build an individual dictionary for
        # that model and then create a Model instance using it.
        for model_num in range(self.num_models):

            model_dict = {}
            for field in all_models_dict.keys():
                model_dict[field] = all_models_dict[field][model_num]

            # Use the correct subclass depending upon the format SAGE wrote in.
            if model_dict["sage_output_format"] == "sage_binary":
                model = SageBinaryModel(model_dict, plot_toggles)
            elif model_dict["sage_output_format"] == "sage_hdf5":
                model = SageHdf5Model(model_dict, plot_toggles)

            model.plot_output_format = plot_output_format

            model.set_cosmology()

            # To be more memory concious, we calculate the required properties on a
            # file-by-file basis. This ensures we do not keep ALL the galaxy data in memory.
            model.calc_properties_all_files(debug=debug)

            all_models.append(model)

        self.models = all_models
        self.plot_toggles = plot_toggles


    def do_plots(self):
        """
        Wrapper method to perform all the plotting for the models.

        Parameters
        ----------

        None.

        Returns
        -------

        None. The plots are saved individually by each method.
        """

        plots.setup_matplotlib_options()

        # Go through all the plot toggles and seach for a plot routine named
        # "plot_<Toggle>".
        for toggle in self.plot_toggles.keys():
            if self.plot_toggles[toggle]:
                method_name = "plot_{0}".format(toggle)

                # If the method doesn't exist, we will hit an `AttributeError`.
                try:
                    getattr(plots, method_name)(self)
                except AttributeError:
                    msg = "Tried to plot '{0}'.  However, no " \
                          "method named '{1}' exists in the 'plots.py' module.\n" \
                          "Check either that your plot toggles are set correctly or add " \
                          "a method called '{1}' to the 'plots.py' module.".format(toggle, \
                          method_name)
                    msg += "\nPLEASE SCROLL UP AND MAKE SURE YOU'RE READING ALL ERROR " \
                           "MESSAGES! THEY'RE EASY TO MISS! :)"
                    raise AttributeError(msg)


if __name__ == "__main__":

    import os

    # We support the plotting of an arbitrary number of models. To do so, simply add the
    # extra variables specifying the path to the model directory and other variables.
    # E.g., 'model1_sage_output_format = ...", "model1_dir_name = ...".
    # `first_file`, `last_file`, `simulation` and `num_tree_files` only need to be
    # specified if using binary output. HDF5 will automatically detect these.
    # `hdf5_snapshot` is only nedded if using HDF5 output.

    model0_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model0_dir_name            = "../output/mini_millennium/"
    model0_file_name           = "model.hdf5"
    model0_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model0_model_label         = "z=0"
    model0_color               = "r"
    model0_linestyle           = "-"
    model0_marker              = "x"
    model0_first_file          = 0  # The files read in will be [first_file, last_file]
    model0_last_file           = 0  # This is a closed interval.
    model0_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model0_hdf5_snapshot       = 63 # Snapshot we're plotting the HDF5 data at.
    model0_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.

    model0_5_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model0_5_dir_name            = "../output/mini_millennium/"
    model0_5_file_name           = "model.hdf5"
    model0_5_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model0_5_model_label         = "z=0.5"
    model0_5_color               = "g"
    model0_5_linestyle           = "-"
    model0_5_marker              = "x"
    model0_5_first_file          = 0  # The files read in will be [first_file, last_file]
    model0_5_last_file           = 0  # This is a closed interval.
    model0_5_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model0_5_hdf5_snapshot       = 48 # Snapshot we're plotting the HDF5 data at.
    model0_5_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.

    model1_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model1_dir_name            = "../output/mini_millennium/"
    model1_file_name           = "model.hdf5"
    model1_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model1_model_label         = "z=1"
    model1_color               = "m"
    model1_linestyle           = "-"
    model1_marker              = "x"
    model1_first_file          = 0  # The files read in will be [first_file, last_file]
    model1_last_file           = 0  # This is a closed interval.
    model1_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model1_hdf5_snapshot       = 40 # Snapshot we're plotting the HDF5 data at.
    model1_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    model2_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model2_dir_name            = "../output/mini_millennium/"
    model2_file_name           = "model.hdf5"
    model2_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model2_model_label         = "z=2"
    model2_color               = "k"
    model2_linestyle           = "-"
    model2_marker              = "x"
    model2_first_file          = 0  # The files read in will be [first_file, last_file]
    model2_last_file           = 0  # This is a closed interval.
    model2_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model2_hdf5_snapshot       = 32 # Snapshot we're plotting the HDF5 data at.
    model2_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    model3_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model3_dir_name            = "../output/mini_millennium/"
    model3_file_name           = "model.hdf5"
    model3_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model3_model_label         = "z=3"
    model3_color               = "y"
    model3_linestyle           = "-"
    model3_marker              = "x"
    model3_first_file          = 0  # The files read in will be [first_file, last_file]
    model3_last_file           = 0  # This is a closed interval.
    model3_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model3_hdf5_snapshot       = 27 # Snapshot we're plotting the HDF5 data at.
    model3_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    model4_5_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model4_5_dir_name            = "../output/mini_millennium/"
    model4_5_file_name           = "model.hdf5"
    model4_5_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model4_5_model_label         = "z=4.5"
    model4_5_color               = "b"
    model4_5_linestyle           = "-"
    model4_5_marker              = "x"
    model4_5_first_file          = 0  # The files read in will be [first_file, last_file]
    model4_5_last_file           = 0  # This is a closed interval.
    model4_5_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model4_5_hdf5_snapshot       = 22 # Snapshot we're plotting the HDF5 data at.
    model4_5_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    model6_2_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model6_2_dir_name            = "../output/mini_millennium/"
    model6_2_file_name           = "model.hdf5"
    model6_2_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model6_2_model_label         = "z=6.2"
    model6_2_color               = "c"
    model6_2_linestyle           = "-"
    model6_2_marker              = "x"
    model6_2_first_file          = 0  # The files read in will be [first_file, last_file]
    model6_2_last_file           = 0  # This is a closed interval.
    model6_2_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model6_2_hdf5_snapshot       = 18 # Snapshot we're plotting the HDF5 data at.
    model6_2_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    model10_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model10_dir_name            = "../output/mini_millennium/"
    model10_file_name           = "model.hdf5"
    model10_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model10_model_label         = "z=10"
    model10_color               = "0.75"
    model10_linestyle           = "-"
    model10_marker              = "x"
    model10_first_file          = 0  # The files read in will be [first_file, last_file]
    model10_last_file           = 0  # This is a closed interval.
    model10_simulation          = "Mini-Millennium"  # Sets the cosmology. Required for "sage_binary".
    model10_hdf5_snapshot       = 12 # Snapshot we're plotting the HDF5 data at.
    model10_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    
    
    # Then extend each of these lists for all the models that you want to plot.
    # E.g., 'dir_names = [model0_dir_name, model1_dir_name, ..., modelN_dir_name]
    sage_output_formats = [model0_sage_output_format, model0_5_sage_output_format, 
                        model1_sage_output_format, model2_sage_output_format, 
                        model3_sage_output_format, model4_5_sage_output_format, 
                        model6_2_sage_output_format, model10_sage_output_format]
    dir_names           = [model0_dir_name, model0_5_dir_name, model1_dir_name, model2_dir_name, 
                        model3_dir_name, model4_5_dir_name, model6_2_dir_name, model10_dir_name]
    file_names          = [model0_file_name, model0_5_file_name, model1_file_name,
                         model2_file_name, model3_file_name, model4_5_file_name,
                        model6_2_file_name, model10_file_name]
    IMFs                = [model0_IMF, model0_5_IMF, model1_IMF, model2_IMF, model3_IMF, model4_5_IMF,
                        model6_2_IMF, model10_IMF]
    model_labels        = [model0_model_label, model0_5_model_label, model1_model_label,
                        model2_model_label, model3_model_label, model4_5_model_label,
                        model6_2_model_label, model10_model_label]
    colors              = [model0_color, model0_5_color, model1_color, model2_color, 
                        model3_color, model4_5_color, model6_2_color, model10_color]
    linestyles          = [model0_linestyle, model0_5_linestyle, model1_linestyle, 
                        model2_linestyle, model3_linestyle, model4_5_linestyle, 
                        model6_2_linestyle, model10_linestyle]
    markers             = [model0_marker, model0_5_marker, model1_marker, model2_marker, 
                        model3_marker, model4_5_marker, model6_2_marker, model10_marker]
    first_files         = [model0_first_file, model0_5_first_file, model1_first_file, 
                        model2_first_file, model3_first_file, model4_5_first_file, 
                        model6_2_first_file, model10_first_file]
    last_files          = [model0_last_file, model0_5_last_file, model1_last_file, 
                        model2_last_file, model3_last_file, model4_5_last_file, 
                        model6_2_last_file, model10_last_file]
    simulations         = [model0_simulation, model0_5_simulation, model1_simulation, 
                        model2_simulation, model3_simulation, model4_5_simulation, 
                        model6_2_simulation, model10_simulation]
    hdf5_snapshots      = [model0_hdf5_snapshot, model0_5_hdf5_snapshot, model1_hdf5_snapshot, 
                        model2_hdf5_snapshot, model3_hdf5_snapshot, model4_5_hdf5_snapshot, 
                        model6_2_hdf5_snapshot, model10_hdf5_snapshot]
    num_tree_files_used = [model0_num_tree_files_used, model0_5_num_tree_files_used, 
                        model1_num_tree_files_used, model2_num_tree_files_used, 
                        model3_num_tree_files_used, model4_5_num_tree_files_used, 
                        model6_2_num_tree_files_used, model10_num_tree_files_used]
    
    # A couple of extra variables...
    plot_output_format    = ".png"
    plot_output_path = "./mini_millennium_hdf5_plots"  # Will be created if path doesn't exist.

    # These toggles specify which plots you want to be made.
    plot_toggles = {"SMF"             : 0,  # Stellar mass function.
                    "BMF"             : 0,  # Baryonic mass function.
                    "GMF"             : 0,  # Gas mass function (cold gas).
                    "BTF"             : 0,  # Baryonic Tully-Fisher.
                    "SFR"             : 0,  # Star formation rate.
                    "SFR_binned"      : 1,  # Star formation rate binned.
                    "sSFR"            : 0,  # Specific star formation rate.
                    "gas_frac"        : 0,  # Fraction of galaxy that is cold gas.
                    "metallicity"     : 0,  # Metallicity scatter plot.
                    "bh_bulge"        : 0,  # Black hole-bulge relationship.
                    "quiescent"       : 0,  # Fraction of galaxies that are quiescent.
                    "bulge_fraction"  : 0,  # Fraction of galaxies that are bulge/disc dominated.
                    "baryon_fraction" : 0,  # Fraction of baryons in galaxy/reservoir.
                    "reservoirs"      : 0,  # Mass in each reservoir.
                    "spatial"         : 0}  # Spatial distribution of galaxies.

    ############################
    ## DON'T TOUCH BELOW HERE ##
    ############################

    model_paths = []
    output_paths = []

    # Determine paths for each model.
    for dir_name, file_name  in zip(dir_names, file_names):

        model_path = "{0}{1}".format(dir_name, file_name)
        model_paths.append(model_path)

        # These are model specific. Used for rare circumstances and debugging.
        output_path = dir_name + "plots/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_paths.append(output_path)

    model_dict = { "sage_output_format"  : sage_output_formats,
                   "model_path"          : model_paths,
                   "output_path"         : output_paths,
                   "IMF"                 : IMFs,
                   "model_label"         : model_labels,
                   "color"               : colors,
                   "linestyle"           : linestyles,
                   "marker"              : markers,
                   "first_file"          : first_files,
                   "last_file"           : last_files,
                   "simulation"          : simulations,
                   "hdf5_snapshot"       : hdf5_snapshots,
                   "num_tree_files_used" : num_tree_files_used}

    # Read in the galaxies and calculate properties for each model.
    results = Results(model_dict, plot_toggles, plot_output_path, plot_output_format,
                      debug=False)
    results.do_plots()

    # Set the error settings to the previous ones so we don't annoy the user.
    np.seterr(divide=old_error_settings["divide"], over=old_error_settings["over"],
              under=old_error_settings["under"], invalid=old_error_settings["invalid"])
