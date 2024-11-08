Code and data to compare steady-state and time-dependent implementations
of WSA-ADAPT-HUXt. This approach is described in:

The importance of boundary evolution for solar-wind modelling, Owens, Barnard 
and Arge, Scientific Reports, doi:XXX, 2024

Requires HUXt to be installed.

Data are provided to reproduce the analysis in the paper. Both scripts should
be run with the load_maps_now flag set to False to use these data. To generate 
data from WSA fits files, set  load_maps_now = True and amend "datadir" 
to point to these files.
