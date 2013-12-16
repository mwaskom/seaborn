test:

	python ipnbdoctest.py *.ipynb

pat = 'Figure at 0x[a-f0-9]*>'
rep = 'Figure at 0xFFFFFFFFF>'
sed = sed s/$(pat)/$(rep)/ | sed s/}\n\r/}/
dir = .

hexstrip:

	cat $(dir)/aesthetics.ipynb | $(sed) > $(dir)/aesthetics_sed.ipynb
	mv $(dir)/aesthetics_sed.ipynb $(dir)/aesthetics.ipynb

	cat $(dir)/linear_models.ipynb | $(sed) > $(dir)/linear_models_sed.ipynb
	mv $(dir)/linear_models_sed.ipynb $(dir)/linear_models.ipynb

	cat $(dir)/plotting_distributions.ipynb | $(sed) > $(dir)/plotting_distributions_sed.ipynb
	mv $(dir)/plotting_distributions_sed.ipynb $(dir)/plotting_distributions.ipynb

	cat $(dir)/timeseries_plots.ipynb | $(sed) > $(dir)/timeseries_plots_sed.ipynb
	mv $(dir)/timeseries_plots_sed.ipynb $(dir)/timeseries_plots.ipynb
