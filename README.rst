EVQ
===

Evolving Vector Quantization for Classification of On-Line Data Streams

Note
====

Inspired by this algorithm and several others, I created a new one
called SEVQ: https://github.com/sylwekczmil/sevq

Inspiration
===========

Implementation done by Sylwester Czmil based on pseudocode and algorithm
description from:

E. Lughofer, "Evolving Vector Quantization for Classification of On-Line
Data Streams," 2008 International Conference on Computational
Intelligence for Modelling Control & Automation, 2008, pp.779-784, doi:
10.1109/CIMCA.2008.47.


Installation
============

.. code:: bash

   # create venv and activate
   # install algorithm
   pip3 evq

Example usage
=============

Training and prediction one sample at a time
                                            

.. code:: python3

   from evq.algorithm import EVQ

   c = EVQ(number_of_classes=2, vigilance=0.1)
   c.partial_fit([-2, -2], 1)
   c.partial_fit([-1, -1], 0)
   c.partial_fit([1, 1], 0)
   c.partial_fit([2, 2], 1)

   print(c.predict([0, 0]))  # 0 
   print(c.predict([3, 3]))  # 1
   print(c.predict([-3, -3]))  # 1

Training and prediction on multiple samples
                                           

.. code:: python3

   from evq.algorithm import EVQ

   c = EVQ(number_of_classes=2, vigilance=0.1)
   c.fit(
       [[-2, -2], [-1, -1], [1, 1], [2, 2]],
       [1, 0, 0, 1],
       epochs=1, permute=False
   )

   print(c.predict([[0, 0], [3, 3], [-3, -3]]))  # [0, 1, 1]
