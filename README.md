# Pile-up emulator

Author: Ole Koenig, Harvard-Smithsonian Center for Astrophysics (ole.koenig@cfa.harvard.edu)

This machine learning model aims to emulate the effect of _photon pile-up_ that occurs in silicon-based X-ray detectors. Loosely speaking, pile-up occurs if a celestial X-ray source is so bright that the detector cannot identify individual X-ray photons anymore. If two photons hit the same pixel during one read-out cycle, it is not possible to distinguish these photons. Instead, the detector assumes that only one photon impacted the pixel, with the energy of both photons assigned to the event. This highly non-linear effect causes distortions in the energy spectrum of the celestial source, making it challenging to understand the data.

The current training data is simulated using the SIXTE simulator (Dauser et al., A&A 630, 66, 2019). Successful application of simulations for understanding pattern and energy pile-up has been shown, among others, in Koenig et al., Nature 605, 248, 2022.
