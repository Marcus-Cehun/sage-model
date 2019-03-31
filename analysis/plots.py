#!/usr/bin/env python
"""
Module to handle creation of plots from the ``SAGE`` model. The passed ``Results`` class
is generated by ``allresults.py``.

Author: Jacob Seiler
"""

import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler
import numpy as np

import observations as obs


def setup_matplotlib_options():
    """
    Set the default plotting parameters.
    """

    matplotlib.rcdefaults()
    plt.rc('axes',
           prop_cycle=(cycler('color', ['k', 'b', 'r', 'g', 'm', '0.5'])),
           labelsize='x-large')
    plt.rc('xtick', labelsize='x-large')
    plt.rc('ytick', labelsize='x-large')
    plt.rc('lines', linewidth='2.0')
    plt.rc('legend', numpoints=1, fontsize='x-large')


def adjust_legend(ax, location="upper right", scatter_plot=0): 
    """
    Adjusts the legend of a specified axis.

    Parameters
    ----------

    ax : ``matplotlib`` axes object
        The axis whose legend we're adjusting

    location : String, default "upper right". See ``matplotlib`` docs for full options
        Location for the legend to be placed.

    scatter_plot : {0, 1}
        For plots involved scattered-plotted data, we adjust the size and alpha of the
        legend points.

    Returns
    -------

    None. The legend is placed directly onto the axis.
    """

    legend = ax.legend(loc=location)
    handles = legend.legendHandles

    legend.draw_frame(False)

    # First adjust the text sizes.
    for t in legend.get_texts():
        t.set_fontsize("medium")

    # For scatter plots, we want to increase the marker size.
    if scatter_plot:
        for handle in handles:
            # We may have lines in the legend which we don't want to touch here.
            if isinstance(handle, matplotlib.collections.PathCollection):
                handle.set_alpha(1.0) 
                handle.set_sizes([10.0])


def plot_SMF(results, plot_sub_populations=False):
    """
    Plots the stellar mass function for the models within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    plot_sub_populations : Boolean, default False
        If ``True``, plots the stellar mass function for red and blue sub-populations.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/1.StellarMassFunction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Go through each of the models and plot. 
    for model in results.models:

        model_label = model.model_label

        # If we only have one model, we will split it into red and blue
        # sub-populations.
        if len(results.models) > 1:
            color = model.color
            ls = model.linestyle
        else:
            color = "k"
            ls = "-"

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # The SMF is normalized by the simulation volume which is in Mpc/h. 
        ax.plot(bin_middles[:-1], model.SMF/model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                color=color, ls=ls, label=model_label + " - All")

        # Be careful to not overcrowd the plot. 
        if results.num_models == 1 or plot_sub_populations:
            ax.plot(bin_middles[:-1], model.red_SMF/model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                    "r:", lw=2, label=model_label + " - Red")
            ax.plot(bin_middles[:-1], model.blue_SMF/model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                    "b:", lw=2, label=model_label + " - Blue")

    # For scaling the observational data, we use the values of the zeroth
    # model.
    zeroth_hubble_h = (results.models)[0].hubble_h
    zeroth_IMF = (results.models)[0].IMF
    ax = obs.plot_smf_data(ax, zeroth_hubble_h, zeroth_IMF) 

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$")

    ax.set_yscale("log", nonposy="clip")

    # Find the models that have the smallest/largest stellar mass bin.
    xlim_min = np.min([model.stellar_mass_bins for model in results.models]) - 0.2
    xlim_max = np.max([model.stellar_mass_bins for model in results.models]) + 0.2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([1.0e-6, 1.0e-1])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/1.StellarMassFunction{1}".format(results.plot_output_path,
                                                       results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_temporal_SMF(temporal_results):
    """
    Plots the evolution of the stellar mass function for the models within the ``TemporalResults``
    class instance.

    Parameters
    ==========

    temporal_results : ``TemporalResults`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``history.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as
    "<temporal_results.plot_output_path>/A.StellarMassFunction<temporal_results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Go through each of the models and plot. 
    for model in temporal_results.models:

        ls = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # Iterate over the snapshots.
        for snap in model.SMF_snaps:
            model_label = "{0} z = {1:.3f}".format(model.model_label, model.redshifts[snap])

            # The SMF is normalized by the simulation volume which is in Mpc/h.
            ax.plot(bin_middles[:-1], model.SMF_dict[snap] / model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                    ls=ls, label=model_label)

    # For scaling the observational data, we use the values of the zeroth
    # model.
    zeroth_IMF = (temporal_results.models)[0].IMF
    ax = obs.plot_temporal_smf_data(ax, zeroth_IMF) 

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$")

    ax.set_yscale("log", nonposy="clip")

    # Find the models that have the smallest/largest stellar mass bin.
    xlim_min = np.min([model.stellar_mass_bins for model in temporal_results.models]) - 0.2
    xlim_max = np.max([model.stellar_mass_bins for model in temporal_results.models]) + 0.2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([1.0e-6, 1.0e-1])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/A.StellarMassFunction{1}".format(temporal_results.plot_output_path,
                                                        temporal_results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_BMF(results):
    """
    Plots the baryonic mass function for the models within the ``Results`` class instance.
    This is the mass function for the stellar mass + cold gas.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/2.BaryonicMassFunction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        ls = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # The MF is normalized by the simulation volume which is in Mpc/h. 
        ax.plot(bin_middles[:-1], model.BMF/model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                color=color, ls=ls, label=model_label + " - All")

    # For scaling the observational data, we use the values of the zeroth
    # model.
    zeroth_hubble_h = (results.models)[0].hubble_h
    zeroth_IMF = (results.models)[0].IMF
    ax = obs.plot_bmf_data(ax, zeroth_hubble_h, zeroth_IMF) 

    ax.set_xlabel(r"$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$")

    ax.set_yscale("log", nonposy="clip")

    # Find the models that have the smallest/largest stellar mass bin.
    xlim_min = np.min([model.stellar_mass_bins for model in results.models]) - 0.2
    xlim_max = np.max([model.stellar_mass_bins for model in results.models]) + 0.2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([1.0e-6, 1.0e-1])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/2.BaryonicMassFunction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_GMF(results):
    """
    Plots the baryonic mass function for the models within the ``Results`` class instance.
    This is the mass function for the cold gas.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/3.GasMassFunction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        ls = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # The MMF is normalized by the simulation volume which is in Mpc/h. 
        ax.plot(bin_middles[:-1], model.GMF/model.volume*pow(model.hubble_h, 3)/model.stellar_bin_width,
                color=color, ls=ls, label=model_label + " - Cold Gas")

    # For scaling the observational data, we use the values of the zeroth
    # model.
    zeroth_hubble_h = (results.models)[0].hubble_h
    obs.plot_gmf_data(ax, zeroth_hubble_h)

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{X}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$")

    ax.set_yscale("log", nonposy="clip")

    # Find the models that have the smallest/largest stellar mass bin.
    xlim_min = np.min([model.stellar_mass_bins for model in results.models]) - 0.2
    xlim_max = np.max([model.stellar_mass_bins for model in results.models]) + 0.2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([1.0e-6, 1.0e-1])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/3.GasMassFunction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)  # Save the figure
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_BTF(results):
    """
    Plots the baryonic Tully-Fisher relationship for the models within the ``Results`` class instance.
    This is the mass function for the cold gas.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/4.BaryonicTullyFisher<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        marker = model.marker

        ax.scatter(model.BTF_vel, model.BTF_mass, marker=marker, s=1,
                   color=color, alpha=0.5, label=model_label + " Sb/c galaxies")

    ax.set_xlim([1.4, 2.6])
    ax.set_ylim([8.0, 12.0])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    ax.set_xlabel(r"$\log_{10}V_{max}\ (km/s)$")
    ax.set_ylabel(r"$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$")

    ax = obs.plot_btf_data(ax) 

    adjust_legend(ax, location="upper left", scatter_plot=1)

    fig.tight_layout()

    output_file = "{0}/4.BaryonicTullyFisher{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()
        

def plot_sSFR(results):
    """
    Plots the baryonic specific star formation rate as a function of stellar mass for the models within the
    ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/5.SpecificStarFormationRate<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        marker = model.marker

        ax.scatter(model.sSFR_mass, model.sSFR_sSFR, marker=marker, s=1, color=color,
                   alpha=0.5, label=model_label)

    # Overplot a dividing line between passive and SF galaxies. 
    w = np.arange(7.0, 13.0, 1.0)
    min_sSFRcut = np.min([model.sSFRcut for model in results.models]) 
    ax.plot(w, w/w*min_sSFRcut, "b:", lw=2.0)

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\log_{10}\ s\mathrm{SFR}\ (\mathrm{yr^{-1}})$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([-16.0, -8.0])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    adjust_legend(ax, scatter_plot=1)

    fig.tight_layout()

    output_file = "{0}/5.SpecificStarFormationRate{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_gas_frac(results):
    """
    Plots the fraction of baryons that are in the cold gas reservoir as a function of
    stellar mass for the models within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/6.GasFraction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        marker = model.marker

        ax.scatter(model.gas_frac_mass, model.gas_frac, marker=marker, s=1, color=color,
                   alpha=0.5, label=model_label + " Sb/c galaxies")

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\mathrm{Cold\ Mass\ /\ (Cold+Stellar\ Mass)}$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([0.0, 1.0])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    adjust_legend(ax, scatter_plot=1)

    fig.tight_layout()

    output_file = "{0}/6.GasFraction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()
        

def plot_metallicity(results):
    """
    Plots the metallicity as a function of stellar mass for the models within the
    ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/7.Metallicity<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        marker = model.marker

        ax.scatter(model.metallicity_mass, model.metallicity, marker=marker, s=1, color=color,
                   alpha=0.5, label=model_label + " galaxies")

    # Use the IMF of the zeroth model to scale the observational results.
    zeroth_IMF = (results.models)[0].IMF
    ax = obs.plot_metallicity_data(ax, zeroth_IMF) 

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$12\ +\ \log_{10}[\mathrm{O/H}]$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([8.0, 9.5])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # Since we're doing a scatter plot, we need to resize the legend points.
    adjust_legend(ax, location="upper right", scatter_plot=1)

    fig.tight_layout()

    output_file = "{0}/7.Metallicity{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()
        

def plot_bh_bulge(results):
    """
    Plots the black-hole bulge relationship for the models within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/8.BlackHoleBulgeRelationship<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        marker = model.marker

        ax.scatter(model.bulge_mass, model.bh_mass, marker=marker, s=1, color=color,
                   alpha=0.5, label=model_label + " galaxies")

    ax = obs.plot_bh_bulge_data(ax) 

    ax.set_xlabel(r"$\log\ M_{\mathrm{bulge}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\log\ M_{\mathrm{BH}}\ (M_{\odot})$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([6.0, 10.0])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    adjust_legend(ax, location="upper right", scatter_plot=1)

    fig.tight_layout()

    output_file = "{0}/8.BlackHoleBulgeRelationship{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()
        

def plot_quiescent(results, plot_sub_populations=False):
    """
    Plots the fraction of galaxies that are quiescent as a function of stellar mass for the
    models within the ``Results`` class instance.  The 'quiescent' cut is defined in
    ``model.py`` as ``sSFRcut``.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    plot_sub_populations : Boolean, default False
        If ``True``, plots the centrals and satellite sub-populations.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/9.QuiescentFraction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        linestyle = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # We will keep the colour scheme consistent, but change the line styles.
        ax.plot(bin_middles[:-1], model.quiescent_galaxy_counts / model.SMF,
                label=model_label + " All", color=color, linestyle="-") 

        if results.num_models == 1 or plot_sub_populations:
            ax.plot(bin_middles[:-1], model.quiescent_centrals_counts / model.centrals_MF,
                    label=model_label + " Centrals", color=color, linestyle="--") 

            ax.plot(bin_middles[:-1], model.quiescent_satellites_counts / model.satellites_MF,
                    label=model_label + " Satellites", color=color, linestyle="-.") 

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stellar}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\mathrm{Quescient\ Fraction}$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([0.0, 1.05])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.10))

    adjust_legend(ax, location="upper left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/9.QuiescentFraction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_bulge_mass_fraction(results, plot_var=False):
    """
    Plots the fraction of the stellar mass that is located in the bulge/disk as a function
    of stellar mass for the models within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    plot_var : Boolean, default False
        If ``True``, plots the variance as shaded regions.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/10.BulgeMassFraction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        linestyle = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.stellar_mass_bins + 0.5 * model.stellar_bin_width

        # Remember we need to average the properties in each bin.
        bulge_mean = model.fraction_bulge_sum / model.SMF
        disk_mean = model.fraction_disk_sum / model.SMF

        # The variance has already been weighted when we calculated it.
        bulge_var = model.fraction_bulge_var
        disk_var = model.fraction_disk_var

        # We will keep the colour scheme consistent, but change the line styles.
        ax.plot(bin_middles[:-1], bulge_mean, label=model_label + " bulge",
                color=color, linestyle="-")
        ax.plot(bin_middles[:-1], disk_mean, label=model_label + " disk",
                color=color, linestyle="--")

        if plot_var:
            ax.fill_between(bin_middles[:-1], bulge_mean+bulge_var, bulge_mean-bulge_var,
                            facecolor=color, alpha=0.25)
            ax.fill_between(bin_middles[:-1], disk_mean+disk_var, disk_mean-disk_var,
                            facecolor=color, alpha=0.25)

    ax.set_xlabel(r"$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\mathrm{Stellar\ Mass\ Fraction}$")

    ax.set_xlim([8.0, 12.0])
    ax.set_ylim([0.0, 1.05])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.10))

    adjust_legend(ax, location="upper left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/10.BulgeMassFraction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_baryon_fraction(results, plot_sub_populations=1):
    """
    Plots the total baryon fraction and the baryon fraction in each reservoir as a
    function of halo mass for the models within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    plot_var : Boolean, default False
        If ``True``, plots the variance as shaded regions.

    Returns
    =======

    None.  The plot will be saved as "<results.plot_output_path>/11.BaryonFraction<results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        linestyle = model.linestyle

        # Set the x-axis values to be the centre of the bins.
        bin_middles = model.halo_mass_bins + 0.5 * model.halo_bin_width

        # Remember we need to average the properties in each bin.
        baryon_mean = model.halo_baryon_fraction_sum / model.fof_HMF

        # We will keep the linestyle constant but change the color. 
        ax.plot(bin_middles[:-1], baryon_mean, label=model_label + " Total",
                color=color, linestyle=linestyle)

        # If we have multiple models, we want to be careful of overcrowding the plot.
        if results.num_models == 1 or plot_sub_populations:
            attrs = ["stars", "cold", "hot", "ejected", "ICS"]
            labels = ["Stars", "Cold", "Hot", "Ejected", "ICS"]
            colors = ["k", "b", "r", "g", "y"]

            for (attr, label, color) in zip(attrs, labels, colors):
                attrname = "halo_{0}_fraction_sum".format(attr) 
                mean = getattr(model, attrname) / model.fof_HMF

                ax.plot(bin_middles[:-1], mean, label=model_label + " " + label,
                        color=color, linestyle=linestyle)

    ax.set_xlabel(r"$\mathrm{Central}\ \log_{10} M_{\mathrm{vir}}\ (M_{\odot})$")
    ax.set_ylabel(r"$\mathrm{Baryon\ Fraction}$")

    # Find the models that have the smallest/largest stellar mass bin.
    xlim_min = np.min([model.halo_mass_bins for model in results.models]) - 0.2
    xlim_max = np.max([model.halo_mass_bins for model in results.models]) + 0.2
    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([0.0, 0.23])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    adjust_legend(ax, location="upper left", scatter_plot=0)

    output_file = "{0}/11.BaryonFraction{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_mass_reservoirs(results):
    """
    Plots the mass in each reservoir as a function of galaxy stellar mass for the models
    within the ``Results`` class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  A plot will be saved as
    "<results.plot_output_path>/12.MassReservoirs<model.model_label><results.plot_output_path>"
    for each model in ``results``.
    """

    # This scatter plot will be messy so we're going to make one for each model.
    for model in results.models:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        model_label = model.model_label
        marker = model.marker

        components = ["StellarMass", "ColdGas", "HotGas", "EjectedMass",
                      "IntraClusterStars"]
        attribute_names = ["stars", "cold", "hot", "ejected", "ICS"]
        labels = ["Stars", "Cold Gas", "Hot Gas", "Ejected Gas", "Intracluster Stars"]

        for (component, attribute_name, label) in zip(components,
                                                             attribute_names, labels):

            attr_name = "reservoir_{0}".format(attribute_name)
            ax.scatter(model.reservoir_mvir, getattr(model, attr_name), marker=marker,
                       s=0.3, label=label)

        ax.set_xlabel(r"$\log\ M_{\mathrm{vir}}\ (M_{\odot})$")
        ax.set_ylabel(r"$\mathrm{Reservoir\ Mass\ (M_{\odot})}$")

        ax.set_xlim([10.0, 14.0])
        ax.set_ylim([7.5, 12.5])

        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

        adjust_legend(ax, location="upper left", scatter_plot=1)

        fig.tight_layout()

        output_file = "{0}/12.MassReservoirs_{1}{2}".format(results.plot_output_path,
                                                            model_label, results.plot_output_format)
        fig.savefig(output_file)
        print("Saved file to {0}".format(output_file))
        plt.close()


def plot_spatial_distribution(results):
    """
    Plots the spatial distribution of the galaxies for the models within the ``Results``
    class instance.

    Parameters
    ==========

    results : ``Results`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``allresults.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  A plot will be saved as
    "<results.plot_output_path>/13.SpatialDistribution<model.model_label><results.plot_output_path>"
    for each model in ``results``.
    """

    fig = plt.figure()

    # 4-panel plot.
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for model in results.models:

        model_label = model.model_label
        color = model.color
        linestyle = model.linestyle
        marker = model.marker

        ax1.scatter(model.x_pos, model.y_pos, marker=marker, s=0.3, color=color,
                    alpha=0.5)
        ax2.scatter(model.x_pos, model.z_pos, marker=marker, s=0.3, color=color,
                    alpha=0.5)
        ax3.scatter(model.y_pos, model.z_pos, marker=marker, s=0.3, color=color,
                    alpha=0.5)

        # The bottom right panel will only contain the legend.
        # For some odd reason, plotting `np.nan` causes some legend entries to not
        # appear. Plot junk and we'll adjust the axis to not show it.
        ax4.scatter(-999, -999, marker=marker, color=color, label=model_label)
        ax4.axis("off")

    ax1.set_xlabel(r"$\mathrm{x}\ [\mathrm{Mpc}/h]$")
    ax1.set_ylabel(r"$\mathrm{y}\ [\mathrm{Mpc}/h]$")

    ax2.set_xlabel(r"$\mathrm{x}\ [\mathrm{Mpc}/h]$")
    ax2.set_ylabel(r"$\mathrm{z}\ [\mathrm{Mpc}/h]$")

    ax3.set_xlabel(r"$\mathrm{y}\ [\mathrm{Mpc}/h]$")
    ax3.set_ylabel(r"$\mathrm{z}\ [\mathrm{Mpc}/h]$")

    # Find the model with the largest box. 
    max_box = np.min([model.box_size for model in results.models]) - 0.5
    buffer = max_box*0.05
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([0.0-buffer, max_box+buffer])
        ax.set_ylim([0.0-buffer, max_box+buffer])

        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

    adjust_legend(ax4, location="upper left", scatter_plot=1)

    # Make sure everything remains nicely layed out.
    fig.tight_layout()

    output_file = "{0}/13.SpatialDistribution{1}".format(results.plot_output_path, results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_spatial_3d(pos, output_file, box_size):
    """
    Plots the 3D spatial distribution of galaxies.

    Parameters
    ==========

    pos : ``numpy`` 3D array with length equal to the number of galaxies
        The position (in Mpc/h) of the galaxies.

    output_file : String
        Name of the file the plot will be saved as.

    Returns
    =======

    None.  A plot will be saved as ``output_file``.
    """

    from mpl_toolkits.mplot3d import Axes3D
    from random import sample

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Generate a subsample if necessary.
    num_gals = len(pos) 
    sample_size = 10000
    if num_gals > sample_size:
        w = sample(list(np.arange(num_gals)), sample_size)
    else:
        w = np.arange(num_gals)

    ax.scatter(pos[w,0], pos[w,1], pos[w,2], alpha=0.5)

    ax.set_xlim([0.0, box_size])
    ax.set_ylim([0.0, box_size])
    ax.set_zlim([0.0, box_size])

    ax.set_xlabel(r"$\mathbf{x \: [h^{-1}Mpc]}$")
    ax.set_ylabel(r"$\mathbf{y \: [h^{-1}Mpc]}$")
    ax.set_zlabel(r"$\mathbf{z \: [h^{-1}Mpc]}$")

    fig.tight_layout()

    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_SFRD(temporal_results):
    """
    Plots the evolution of the star formation rate density for the models within the
    ``TemporalResults`` class instance.

    Parameters
    ==========

    temporal_results : ``TemporalResults`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``history.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as
    "<temporal_results.plot_output_path>/B.History-SFR-Density<temporal_results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in temporal_results.models:

        label = model.model_label
        color = model.color
        linestyle = model.linestyle

        # The SFRD is in a dictionary. Pull it out into a array for plotting.
        SFRD = np.array([model.SFRD_dict[snap] for snap in model.SFRD_dict.keys()])
        ax.plot(model.redshifts[model.density_snaps], np.log10(SFRD / model.volume*pow(model.hubble_h, 3)),
                label=label, color=color, ls=linestyle)

    ax = obs.plot_sfrd_data(ax) 

    ax.set_xlabel(r"$\mathrm{redshift}$")
    ax.set_ylabel(r"$\log_{10} \mathrm{SFR\ density}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$")

    ax.set_xlim([0.0, 8.0])
    ax.set_ylim([-3.0, -0.4])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/B.History-SFR-Density{1}".format(temporal_results.plot_output_path,
                                                        temporal_results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()


def plot_SMD(temporal_results):
    """
    Plots the evolution of the stellar mass density for the models within the ``TemporalResults``
    class instance.

    Parameters
    ==========

    temporal_results : ``TemporalResults`` class instance
        Class instance that contains the calculated properties for all the models.  The
        class is defined in the ``history.py`` with the individual ``Model`` classes
        and properties defined and calculated in the ``model.py`` module.

    Returns
    =======

    None.  The plot will be saved as
    "<temporal_results.plot_output_path>/C.History-stellar-mass-Density<temporal_results.plot_output_path>"
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model in temporal_results.models:

        label = model.model_label
        color = model.color
        linestyle = model.linestyle

        # The SMD is in a dictionary. Pull it out into a array for plotting.
        SMD = np.array([model.SMD_dict[snap] for snap in model.SMD_dict.keys()])
        ax.plot(model.redshifts[model.density_snaps], np.log10(SMD / model.volume * pow(model.hubble_h, 3)),
                label=label, color=color, ls=linestyle)

    # For scaling the observational data, we use the values of the zeroth
    # model.
    zeroth_IMF = (temporal_results.models)[0].IMF
    ax = obs.plot_smd_data(ax, zeroth_IMF)

    ax.set_xlabel(r"$\mathrm{redshift}$")
    ax.set_ylabel(r'$\log_{10}\ \phi\ (M_{\odot}\ \mathrm{Mpc}^{-3})$')

    ax.set_xlim([0.0, 4.2])
    ax.set_ylim([6.5, 9.0])

    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    adjust_legend(ax, location="lower left", scatter_plot=0)

    fig.tight_layout()

    output_file = "{0}/C.History-stellar-mass-Density{1}".format(temporal_results.plot_output_path,
                                                                 temporal_results.plot_output_format)
    fig.savefig(output_file)
    print("Saved file to {0}".format(output_file))
    plt.close()
