#!/usr/bin/env python
"""
Handles plotting properties over multiple redshifts.  Multiple models can be placed onto
the plots by extending the variables in the ``__main__`` function call.

To add your own data format, create a subclass module (e.g., ``sage_binary.py``) and add an
option to ``Results.__init__``.  This subclass module needs methods ``set_cosmology()``,
``determine_num_gals()``, ``read_gals()`` and ``update_redshift()``.

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

try:
    from tqdm import tqdm
except ImportError:
    print("Package 'tqdm' not found. Not showing pretty progress bars :(")
else:
    pass

import numpy as np

# Sometimes we divide a galaxy that has zero mass (e.g., no cold gas). Ignore these
# warnings as they spam stdout.
old_error_settings = np.seterr()
np.seterr(all="ignore")


class TemporalResults:
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

            print("Processing data for model {0}".format(model.model_label))

            # We may be plotting the density at all snapshots...
            if model.density_redshifts == -1:
                model.density_redshifts = model.redshifts

            model.plot_output_format = plot_output_format
            model.set_cosmology()

            # The SMF and the Density plots may have different snapshot requirements.
            # Find the snapshots that most closely match the requested redshifts.
            model.SMF_snaps = [(np.abs(model.redshifts - SMF_redshift)).argmin() for
                              SMF_redshift in model.SMF_redshifts]
            model.properties["SMF_dict"] = {}

            model.density_snaps = [(np.abs(model.redshifts - density_redshift)).argmin() for
                                  density_redshift in model.density_redshifts]
            model.properties["SFRD_dict"] = {}
            model.properties["SMD_dict"] = {}

            # We'll need to loop all snapshots, ignoring duplicates.
            snaps_to_loop = np.unique(model.SMF_snaps + model.density_snaps)

            # Use a tqdm progress bar if possible.
            try:
                snap_iter = tqdm(snaps_to_loop, unit="Snapshot")
            except NameError:
                snap_iter = snaps_to_loop
            ###################
            # Here we calculate just model 0 in order to read in galaxies of
            # redshift 0.
            
            # Update to snapshot 63 to make sure we are only getting z=0 galaxies.
            model.update_snapshot(63)
            
            #model_dict = {}
            # We only want to calculate the GalaxyID_list, so set toggle to 1
            model.plot_toggles = {"GalaxyID_List" :1}

            # To be more memory concious, we calculate the required properties on a
            # file-by-file basis. This ensures we do not keep ALL the galaxy data in memory.
            # Calculate model properties for model 0
            model.calc_properties_all_files(close_file=False, debug=False, IDs_to_Process=None)
            # Pass calculated GalaxyID_list to IDs_to_Process or set
            # to None and comment out the passing if you wish to plot all galaxies.
            IDs_to_Process = model.properties["GalaxyID_List"] 
            print("IDs:",IDs_to_Process)
            ################
            # we need to set plot_toggles back to the full set we wish to plot.
            model.plot_toggles = plot_toggles
            
            for snap in snap_iter:

                # Reset the tracking.
                model.properties["SMF"] = np.zeros(len(model.mass_bins)-1, dtype=np.float64)
                model.properties["SFRD"] = 0.0
                model.properties["SMD"] = 0.0

                # Update the snapshot we're reading from. Subclass specific.
                model.update_snapshot(snap)

                # Calculate all the properties. Keep the HDF5 file open always.
                model.calc_properties_all_files(close_file=False, use_pbar=False, debug=False, 
                                                IDs_to_Process=IDs_to_Process)

                # We need to place the SMF inside the dictionary to carry through.
                if snap in model.SMF_snaps:
                    model.properties["SMF_dict"][snap] = model.properties["SMF"]

                # Same with the densities.
                if snap in model.density_snaps:

                    # It's slightly wasteful here because the user may only want the SFRD
                    # and not the SMD. However the wasted compute time is neglible
                    # compared to the time taken to read the galaxies + compute the
                    # properties.
                    model.properties["SFRD_dict"][snap] = model.properties["SFRD"]
                    model.properties["SMD_dict"][snap] = model.properties["SMD"]

            all_models.append(model)

            # If we used a HDF5 file, close it.
            try:
                self.close_file()
            except AttributeError:
                pass

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


if __name__ == '__main__':

    import os

    # We support the plotting of an arbitrary number of models. To do so, simply add the
    # extra variables specifying the path to the model directory and other variables.
    # E.g., 'model1_sage_output_format = ...", "model1_dir_name = ...".
    # `first_file`, `last_file`, `simulation` and `num_tree_files` only need to be
    # specified if using binary output. HDF5 will automatically detect these.

    model0_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model0_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model0_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model0_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model0_dir_name            = "../output/mini_millennium"
    model0_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model0_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model0_model_label         = "8.0 mass cut"
    model0_color               = "c"
    model0_linestyle           = "-"
    model0_marker              = "x"
    model0_first_file          = 0  # The files read in will be [first_file, last_file]
    model0_last_file           = 0  # This is a closed interval.
    model0_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model0_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model0_mass_cut            = [8.0] # Stellar Mass Cut for this model.


    model1_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model1_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model1_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model1_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model1_dir_name            = "../output/mini_millennium"
    model1_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model1_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model1_model_label         = "8.5 mass cut"
    model1_color               = "c"
    model1_linestyle           = "-"
    model1_marker              = "x"
    model1_first_file          = 0  # The files read in will be [first_file, last_file]
    model1_last_file           = 0  # This is a closed interval.
    model1_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model1_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model1_mass_cut            = [8.5] # Stellar Mass Cut for this model.

    model2_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model2_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model2_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model2_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model2_dir_name            = "../output/mini_millennium"
    model2_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model2_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model2_model_label         = "9.0 mass cut"
    model2_color               = "c"
    model2_linestyle           = "-"
    model2_marker              = "x"
    model2_first_file          = 0  # The files read in will be [first_file, last_file]
    model2_last_file           = 0  # This is a closed interval.
    model2_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model2_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model2_mass_cut            = [9.0] # Stellar Mass Cut for this model.

    model3_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model3_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model3_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model3_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model3_dir_name            = "../output/mini_millennium"
    model3_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model3_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model3_model_label         = "9.5 mass cut"
    model3_color               = "c"
    model3_linestyle           = "-"
    model3_marker              = "x"
    model3_first_file          = 0  # The files read in will be [first_file, last_file]
    model3_last_file           = 0  # This is a closed interval.
    model3_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model3_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model3_mass_cut            = [9.5] # Stellar Mass Cut for this model.

    model4_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model4_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model4_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model4_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model4_dir_name            = "../output/mini_millennium"
    model4_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model4_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model4_model_label         = "10.0 mass cut"
    model4_color               = "c"
    model4_linestyle           = "-"
    model4_marker              = "x"
    model4_first_file          = 0  # The files read in will be [first_file, last_file]
    model4_last_file           = 0  # This is a closed interval.
    model4_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model4_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model4_mass_cut            = [10.0] # Stellar Mass Cut for this model.

    model5_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model5_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model5_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model5_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model5_dir_name            = "../output/mini_millennium"
    model5_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model5_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model5_model_label         = "10.5 mass cut"
    model5_color               = "c"
    model5_linestyle           = "-"
    model5_marker              = "x"
    model5_first_file          = 0  # The files read in will be [first_file, last_file]
    model5_last_file           = 0  # This is a closed interval.
    model5_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model5_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model5_mass_cut            = [10.5] # Stellar Mass Cut for this model.

    model6_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model6_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model6_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model6_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model6_dir_name            = "../output/mini_millennium"
    model6_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model6_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model6_model_label         = "11.0 mass cut"
    model6_color               = "c"
    model6_linestyle           = "-"
    model6_marker              = "x"
    model6_first_file          = 0  # The files read in will be [first_file, last_file]
    model6_last_file           = 0  # This is a closed interval.
    model6_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model6_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model6_mass_cut            = [11.0] # Stellar Mass Cut for this model.

    model7_SMF_z               = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the stellar mass function at.
                                 # Will search for the closest simulation redshift.
    model7_density_z           = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.2, 10]  # Redshifts you wish to plot the evolution of
                                     # densities at. Set to -1 for all redshifts.
    model7_alist_file          = "../input/mini_millennium/trees/millennium.a_list"
    model7_sage_output_format  = "sage_hdf5"  # Format SAGE output in. "sage_binary" or "sage_hdf5".
    model7_dir_name            = "../output/mini_millennium"
    model7_file_name           = "model.hdf5"  # If using "sage_binary", doesn't have to end in "_zX.XXX"
    model7_IMF                 = "Chabrier"  # Chabrier or Salpeter.
    model7_model_label         = "11.5 mass cut"
    model7_color               = "c"
    model7_linestyle           = "-"
    model7_marker              = "x"
    model7_first_file          = 0  # The files read in will be [first_file, last_file]
    model7_last_file           = 0  # This is a closed interval.
    model7_simulation          = "Mini-Millennium"  # Sets the cosmology for "sage_binary".
    model7_num_tree_files_used = 8  # Number of tree files processed by SAGE to produce this output.
    model7_mass_cut            = [11.5] # Stellar Mass Cut for this model.

    # Then extend each of these lists for all the models that you want to plot.
    # E.g., 'dir_names = [model0_dir_name, model1_dir_name, ..., modelN_dir_name]
    SMF_zs               = [model0_SMF_z, model1_SMF_z, model2_SMF_z, model3_SMF_z,
                            model4_SMF_z, model5_SMF_z, model6_SMF_z, model7_SMF_z]
    density_zs           = [model0_density_z, model1_density_z, model2_density_z, model3_density_z,
                            model4_density_z, model5_density_z, model6_density_z, model7_density_z]
    alist_files          = [model0_alist_file, model1_alist_file, model2_alist_file, model3_alist_file,
                            model4_alist_file, model5_alist_file, model6_alist_file, model7_alist_file]
    sage_output_formats  = [model0_sage_output_format, model1_sage_output_format, model2_sage_output_format, model3_sage_output_format,
                            model4_sage_output_format, model5_sage_output_format, model6_sage_output_format, model7_sage_output_format]
    dir_names            = [model0_dir_name, model1_dir_name, model2_dir_name, model3_dir_name,
                            model4_dir_name, model5_dir_name, model6_dir_name, model7_dir_name]
    file_names           = [model0_file_name, model1_file_name, model2_file_name, model3_file_name,
                            model4_file_name, model5_file_name, model6_file_name, model7_file_name]
    IMFs                 = [model0_IMF, model1_IMF, model2_IMF, model3_IMF,
                            model4_IMF, model5_IMF, model6_IMF, model7_IMF]
    model_labels         = [model0_model_label, model1_model_label, model2_model_label, model3_model_label,
                            model4_model_label, model5_model_label, model6_model_label, model7_model_label]
    colors               = [model0_color, model1_color, model2_color, model3_color,
                            model4_color, model5_color, model6_color, model7_color]
    linestyles           = [model0_linestyle, model1_linestyle, model2_linestyle, model3_linestyle,
                            model4_linestyle, model5_linestyle, model6_linestyle, model7_linestyle]
    markers              = [model0_marker, model1_marker, model2_marker, model3_marker,
                            model4_marker, model5_marker, model6_marker, model7_marker]
    first_files          = [model0_first_file, model1_first_file, model2_first_file, model3_first_file,
                            model4_first_file, model5_first_file, model6_first_file, model7_first_file]
    last_files           = [model0_last_file, model1_last_file, model2_last_file, model3_last_file,
                            model4_last_file, model5_last_file, model6_last_file, model7_last_file]
    simulations          = [model0_simulation, model1_simulation, model2_simulation, model3_simulation,
                            model4_simulation, model5_simulation, model6_simulation, model7_simulation]
    num_tree_files_used  = [model0_num_tree_files_used, model1_num_tree_files_used, model2_num_tree_files_used, model3_num_tree_files_used,
                            model4_num_tree_files_used, model5_num_tree_files_used, model6_num_tree_files_used, model7_num_tree_files_used]
    mass_cuts            = [model0_mass_cut, model1_mass_cut, model2_mass_cut, model3_mass_cut,
                            model4_mass_cut, model5_mass_cut, model6_mass_cut, model7_mass_cut]

    # A couple of extra variables...
    plot_output_format    = ".png"
    plot_output_path = "./mini_millennium_hdf5_plots"  # Will be created if path doesn't exist.

    # These toggles specify which plots you want to be made.
    plot_toggles = {"SFR_cut"         : 1,  # Star formation rate for a given z=0 mass bin.
                    "SMF"             : 1,  # Stellar mass function at specified redshifts.
                    "SFRD"            : 1,  # Star formation rate density at specified snapshots. 
                    "SMD"             : 0}  # Stellar mass density at specified snapshots. 

    ############################
    ## DON'T TOUCH BELOW HERE ##
    ############################

    model_paths = []
    output_paths = []

    # Determine paths for each model.
    for dir_name, file_name  in zip(dir_names, file_names):

        model_path = "{0}/{1}".format(dir_name, file_name)
        model_paths.append(model_path)

        # These are model specific. Used for rare circumstances and debugging.
        output_path = dir_name + "plots/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_paths.append(output_path)

    model_dict = { "SMF_redshifts"       : SMF_zs,
                   "density_redshifts"   : density_zs,
                   "alist_file"          : alist_files,
                   "sage_output_format"  : sage_output_formats,
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
                   "num_tree_files_used" : num_tree_files_used,
                   "mass_cuts"            : mass_cuts}

    # Read in the galaxies and calculate properties for each model.
    results = TemporalResults(model_dict, plot_toggles, plot_output_path, plot_output_format,
                              debug=False)
    results.do_plots()

    # Set the error settings to the previous ones so we don't annoy the user.
    np.seterr(divide=old_error_settings["divide"], over=old_error_settings["over"],
              under=old_error_settings["under"], invalid=old_error_settings["invalid"])
