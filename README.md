Ensverif is a Python library that contains functions to assess the quality of ensemble forecasts or simulations. Those functions are:

- crps
- crps\_hersbach_decomposition
- logscore
- rankhist
- reliability

Those functions were initially coded in Matlab during my MSc and PhD degree. They were further improved over time, by my students (especially Rachel Bazile) and myself. 

If you don't want to use pip, the library is also available on Github, in Matlab and Python, here https://github.com/TheDroplets

Authors: Marie-Amélie Boucher, Rachel Bazile, Konstantin Ntokas and Alireza Amani

Contact: marie-amelie.boucher@usherbrooke.ca

Main changes in that release: 
- Place all the functions in one single module instead of one function per module, thus facilitating the call to each function and the use of this module:

- Replace np.mean by np.nanmean in the CRPS function
- Correct spelling mistakes
- Correct convention mistakes (not perfect yet, but better)
- Improve documentation (more concise + adding an example on how to use the module)