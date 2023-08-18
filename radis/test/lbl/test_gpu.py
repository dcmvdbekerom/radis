# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 13:34:42 2020

@author: pankaj

------------------------------------------------------------------------

"""
import numpy as np
import pytest

from radis import SpectrumFactory, get_residual
from radis.misc.printer import printm
from radis.test.utils import getTestFile

T = 1000
fixed_conditions = {
    "path_length": 1,
    "wmin": 2284,
    "wmax": 2285,
    "pressure": 0.1,
    "wstep": 0.001,
    "mole_fraction": 0.01,  # until self and air broadening is implemented
}
ignored_warning = {
    "MissingSelfBroadeningWarning": "ignore",
    "NegativeEnergiesWarning": "ignore",
    "HighTemperatureWarning": "ignore",
    "GaussianBroadeningWarning": "ignore",
}


def test_eq_spectrum_emulated_gpu(emulate=True, plot=False, *args, **kwargs):

    print("Emulate: ", emulate)

    sf = SpectrumFactory(
        **fixed_conditions,
        warnings=ignored_warning,
    )
    sf._broadening_method = "fft"
    sf.load_databank(
        path=getTestFile("cdsd_hitemp_09_fragment.txt"),
        format="cdsd-4000",
        parfuncfmt="hapi",
    )

    s_cpu = sf.eq_spectrum(Tgas=T, name="CPU")
    s_gpu = sf.eq_spectrum_gpu(
        Tgas=T, emulate=emulate, name="GPU (emulate)" if emulate else "GPU"
    )
    s_cpu.crop(wmin=2284.2, wmax=2284.8)  # remove edge lines
    s_gpu.crop(wmin=2284.2, wmax=2284.8)
    if plot:
        s_cpu.compare_with(s_gpu, spectra_only=True, plot=plot)
    assert get_residual(s_cpu, s_gpu, "abscoeff") < 1.4e-5
    assert get_residual(s_cpu, s_gpu, "radiance_noslit") < 7.3e-6
    assert get_residual(s_cpu, s_gpu, "transmittance_noslit") < 1.4e-5


@pytest.mark.needs_cuda
def test_eq_spectrum_gpu(plot, *args, **kwargs):
    test_eq_spectrum_emulated_gpu(emulate=False, plot=plot, *args, **kwargs)


@pytest.mark.needs_cuda
@pytest.mark.fast
def test_single_SF_4_several_spectra():
    """
    This function tests if a single SpectrumFactory can be used several times to calculate a spectrum.

    Returns
    -------
    None

    """
    sf = SpectrumFactory(
        **fixed_conditions,
        warnings=ignored_warning,
    )

    sf.load_databank(
        path=getTestFile("cdsd_hitemp_09_fragment.txt"),
        format="cdsd-4000",
        parfuncfmt="hapi",
    )
    #%% Pure CPU
    s1_cpu = sf.eq_spectrum(
        Tgas=1000.0,  # K
    )
    integral_CPU = s1_cpu.get_integral("absorbance")
    #%% Spectrum 1
    s1_gpu = sf.eq_spectrum_gpu(
        Tgas=1000.0,  # K
        emulate=False,  # runs on GPU
    )
    integral_GPU = s1_gpu.get_integral("absorbance")

    assert abs(1 - integral_CPU / integral_GPU) < 1e-5
    #%% Spectrum 2
    s2_gpu = sf.eq_spectrum_gpu(
        Tgas=1100.0,  # K
        emulate=False,  # runs on GPU
    )
    assert not (get_residual(s2_gpu, s1_gpu, "abscoeff") < 1.4e-5)


def test_diluent_broadening():
    sf = SpectrumFactory(**fixed_conditions, warnings=ignored_warning, molecule="CO")
    sf.fetch_databank("hitran")

    s1 = sf.eq_spectrum_gpu(
        Tgas=300, emulate=False, diluent={"air": 0.1, "He": 0.89}  # K  # runs on GPU
    )
    s2 = sf.eq_spectrum_gpu(
        Tgas=300, emulate=False, diluent={"air": 0.99}  # K  # runs on GPU
    )

    # the integral of the absorbance should be the same
    assert np.isclose(s1.get_integral("absorbance"), s2.get_integral("absorbance"))

    # the broadening should NOT be the same
    assert not (get_residual(s1, s2, "abscoeff") < 1.4e-5)


# --------------------------
if __name__ == "__main__":

    printm("Testing GPU spectrum calculation:", pytest.main(["test_gpu.py"]))
