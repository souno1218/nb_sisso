# nb_sisso
This project is a numba implementation so to speak of [SISSO](https://github.com/rouyang2017/SISSO) developed by [rouyang2017].   
Originally, the goal was to reinvent the wheel and gain a better understanding of the above project.   
It is also not a complete re-implementation of the original code, as I myself cannot read fortran code and have therefore read very little of the original code.   
Also, it is implemented separately in SIS and SO, so the results of each need to be connected when used (check `info/how_to_use.ipynb`).   
As my own research is on classification problems and I have not had the opportunity to deal with regression problems,   
 the use of the indicator in regression problems requires a change in the indicator by the USER.   
However, numba is no different from python (numpy) if you just want to read it, and many people can write and read it in python, so I am releasing it as a reference.   
It is also possible to create new indicators (replacing convex hull in classification problems) if you can write python.   

For more information about the original project, you can visit [SISSO](https://github.com/rouyang2017/SISSO).

## Getting Started
### Prerequisites
Requires pymatgen,numpy,pandas. If not, installation is automatic.

### Installing
First, activate the virtual environment if it is separated by conda.
```
#examples
conda activate myenv
```
Download and Installation
```
pip install git+https://github.com/souno1218/nb_sisso.git
```

## Running
See how_to_use.ipynb in the info for execution

## Differences from the Original Project
This project differs from [SISSO](https://github.com/rouyang2017/SISSO) in the following ways:
- Rewritten in python,numpy,numba
- Incomplete and dirty code
- Ease of modification due to being written in python.

## Built With
* [numba](https://numba.pydata.org) - main code
* [numpy](https://numpy.org) - Used for various calculations
* [numba_progress](https://github.com/conda-forge/numba-progress-feedstock) - Display of progress bar

## Authors
* **河野 颯之介(Sonosuke Kono)**

## License
This project is licensed under Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.   

## Acknowledgements
This project is based on [SISSO](https://github.com/rouyang2017/SISSO), originally developed by [Original Author(s)]. The original project is licensed under the Apache License 2.0, and a copy of the license can be found [here](http://www.apache.org/licenses/LICENSE-2.0).

Portions of this project are modifications based on work created and shared by the [rouyang2017] under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

## Finally.
Special thanks to the developers of [SISSO](https://github.com/rouyang2017/SISSO) for their amazing work, which served as the foundation for this project.   

I am Japanese and had never used GitHub until I wrote this.   
I use Deepl because I am not good at English.   
This ReedMe is also written with reference to the following page.   
https://gist.github.com/PurpleBooth/109311bb0361f32d87a2   
