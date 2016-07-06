# Copyright (c) 2016 Johan Berglund
# FWQPBO is distributed under the terms of the GNU General Public License
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

ABOUT
-------------------------------------------------------------------------------
FQPBO v1.0 is a command-line tool for MRI fat-water separation based on QPBO
graph cuts. Input paramaters are provided by human readable configuration files. 
See example configuration files provided with FWQPBO for details. The core 
algorithm is implemented in c++ and compiled into a windows 32-bit DLL file 
which is called by a Python wrapper. 

HOW TO USE
-------------------------------------------------------------------------------
To use the command-line tool, see example file test.bat
To use as a Python script, see example file demo.py (requires downloading data
for ISMRM 2012 challenge from: http://challenge.ismrm.org/node/14)

HOW TO CITE
-------------------------------------------------------------------------------
Berglund J. "Multi-scale graph cut algorithm for water-fat separation". In: 
Proceedings of the 23:rd Annual Meeting of ISMRM, Toronto, Canada, 2015. p 3653.

DEPENDENCIES
-------------------------------------------------------------------------------
FWQPBO was written in Python 3.5.1, using modules SciPy 0.17.0, pyDicom 0.9.9.
The C++ implementation uses Eigen 3.2.2 (http://eigen.tuxfamily.org/) for linear
algebra, an image class written by Pedro Felzenszwalb
(https://github.com/cvjena/Felzenszwalb-Segmentation/blob/master/image.h), and
QPBO v1.4 by Vladimir Kolmogorov for the graph cuts
(http://pub.ist.ac.at/~vnk/software/QPBO-v1.4.src.zip)

CONTACT INFORMATION
-------------------------------------------------------------------------------
Johan Berglund, Ph.D.
Dept. of Medical Physics
Karolinska University Hospital, 
Stockholm, Sweden
johan.berglund@karolinska.se