# CaliCo
#### Calibrator for Cosmology

A Python-based software package for calibrating data from radio interferometric telescopes.
CaliCo is developed and maintained by Ruby Byrne (rbyrne@caltech.edu).

## Approach

* **Modularity**: CaliCo is designed to operate as one component of a modular data processing pipeline. It is intended to be paired with a visibility simulator that generates model visibilities for calibration, with flagging software that defines contaminated data, and with imaging software or other downstream processing modules. It provides only the calibration optimization step.
* **Flexibility**: CaliCo supports a variety of statistical frameworks for calibration and is designed to easily accommodate future extensions that will add support for additional calibration frameworks. It is instrument agnostic.
* **Optimization with Newton's Method**: CaliCo supports Newton's Method optimization with explicit, analytical first and second derivative functions. This helps make optimization efficient and accurate. It uses an off-the-shelf optimizer from scipy.

## Dependencies

### Software Dependencies

* pyuvdata ([[https://github.com/RadioAstronomySoftwareGroup/pyuvdata]])
* numpy
* scipy
* astropy

### Other Requirements

* Visibility data in a pyuvdata-readable format (see [[https://pyuvdata.readthedocs.io/]] for details)
* Model visibilities in a pyuvdata-readable format, generated with the visibility simulator of your choice. Some options for visibility simulators include:
    * pyuvsim ([[https://github.com/RadioAstronomySoftwareGroup/pyuvsim]], see [Lanman et al. 2019](https://doi.org/10.21105/joss.01234))
    * matvis ([[https://github.com/HERA-Team/matvis]], see [Kittiwisit et al. 2023](https://doi.org/10.48550/arXiv.2312.09763))
    * fftvis ([[https://github.com/tyler-a-cox/fftvis]])
    * FHD ([[https://github.com/EoRImaging/FHD]], see [Sullivan et al. 2013](https://doi.org/10.1088/0004-637X/759/1/17) and [Barry et al. 2019](https://doi.org/10.1017/pasa.2019.21))
* Data flags, generated with the flagging software of your choice and saved to the data file. Some options for flagging software include:
    * aoflagger
    * SSINS ([[https://github.com/mwilensky768/SSINS]], see [Wilensky et al. 2019](https://doi.org/10.1088/1538-3873/ab3cad))

## Documentation

See Calibration_Cost_Functions_and_Derivatives.pdf for detailed documentation of the mathematical formalism used in CaliCo. See also [Byrne et al. 2021](https://doi.org/10.1093/mnras/stab647), "A Unified Calibration Framework for 21 cm Cosmology", and [Byrne 2023](https://doi.org/10.3847/1538-4357/acac95), "Delay-weighted Calibration: Precision Calibration for 21 cm Cosmology with Resilience to Sky Model Error".