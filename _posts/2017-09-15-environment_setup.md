---
layout: post
published: true
title: "Environment Setup"
headline: "Preparation makes perfect"
mathjax: true
featured: true
categories: curriculum

comments: true
---

<p>
  Before getting into the actual programming lessons, we first need to set up our
  coding environment.
</p>

<p>
  <b>Python</b> is the most popular tool for machine learning, and it is what
  we will be using throughout this curriculum. Now you might be wondering: why
  use Python, a high level programming language, for machine learning?
  Wouldn't things run faster if we used C++?
</p>

<p>
  It turns out, we can run our machine learning programs in Python just as fast
  as we could in C++ -- and the simplicity of Python makes it an attractive
  choice. In a typical deep learning program, all of the
  computational burden will be from large matrix multiplications. There is a
  library for Python called <b>NumPy</b>, a scientific computing library that offers a
  Python interface for low level, fast math operations, especially with
  matrices and other linear algebra concepts. The same is true for <b>TensorFlow</b>:
  the deep learning package that we will be using throughout
  most of our workshops. TensorFlow provides a Python
  interface for fast deep learning operations that can be performed on either the CPU
  or GPU (the GPU is a <b>lot</b> faster at matrix multiplications than the
  CPU, making it valuable for deep learning applications). Finally, we will
  also install <b>scikit-learn</b>: a general machine learning package for Python.
</p>

<p>
  These instructions will walk you through what you will need to install. In
  summary, we will be installing Anaconda, TensorFlow and scikit-learn, all using
  Python 3.5. These instructions should work for both Windows and any Unix
  system. (However, I know Windows commonly has problems installing this
  software. I highly recommend using any Unix system if you have one, not
  including a VM).
</p>

1. A lot of the functionality in Python comes from external packages. **Anaconda**
   is a package manager that we will use to manage our environments and
   versions of Python. Download the Python 3.6 version of Anaconda from <a
   href='https://www.anaconda.com/download/'>here</a>. Note that if you currently
   have Python installed, you may need to uninstall your existing installation
   first in order to avoid any conflicts during the Anaconda setup. Lastly, if
   you are on Windows, make sure you check the box "Add Anaconda to my PATH 
   environment variable" during setup. It is not the default option, so be
   sure not to miss it!
2. Check that anaconda is installed by running in your terminal <code class='language-bash'>conda info</code>. If you are on Windows and getting an error, see Raghav's comment
   below for a possible fix.
3. Create your conda environment. This will specify a certain version of Python to use, and will act as separated container (apart from your root installation) for all of your Python packages to exist. Run <code
   class='language-bash'>conda create -n caispp python=3.5</code> in your
   terminal. This command specifies to use Python 3.5 for our virtual environment,
   and names our environment 'caispp'.
4. Activate your environment. This tells your terminal session to use the
   version of Python and the packages in the conda environment. This is done
   through <code class='language-bash'>activate caispp</code> on Windows or
   <code class='language-bash'>source activate caispp</code> on Unix systems.
   You should see your prompt change with the name of the environment to the
   left of the input line. Make sure this environment is activated while doing
   the remaining steps.
5. Make sure that **pip** is installed by running <code class='language-bash'>pip
   -v</code>. Pip is an easy-to-use package manager built for Python, and we will use it to install several of the packages we will need in the future.
   If pip is not installed, follow the instructions <a
   href='https://pip.pypa.io/en/stable/installing/'>here</a>.
6. Install **TensorFlow**. For Windows, run <code class='language-bash'>pip install --ignore-installed --upgrade tensorflow</code>. For Unix based systems, just run <code class='language-bash'>pip install tensorflow</code>.
7. Check that TensorFlow was actually installed. Start up a Python instance in terminal with
   the command:
   <code class='language-bash'>python</code>. When in the Python instance, import TensorFlow using: <code class='language-python'>import tensorflow as tf</code>. At this point,
   you might see some warning logs or other messages, but as long as it didn't
   give an error, you are good to go! You can now exit out of Python by entering `ctrl-d` or `ctrl-c`. (Oome machines use one or the other, so try both.)
8. Install **scikit-learn**. Scikit-learn is on `conda`, so we just need to enter into the command line: <code class='language-bash'>conda install scikit-learn</code>. Once again, test what we just installed. Create another Python instance in terminal, and try importing the package: <code class='language-python'>import sklearn</code>.
9. NumPy should have been installed as a dependency of the other packages, but it may be a good idea to
   ensure that numpy is also installed and working. Go ahead and launch another Python instance and
   type the following code: <code class='language-python'>import numpy as np</code>. Again, if you didn't get an error, then that means numpy was installed correctly.
10. Final installations: exit out of your Python instance, and run these commands into the command line: <code class='language-bash'>conda install nb_conda</code> (to make our conda environment compatible with [Jupyter Notebooks](http://jupyter.org/)), <code class='language-bash'>pip install matplotlib</code> (a plotting library for Python), <code class='language-bash'>pip install pandas</code> (a data table library), and <code class='language-bash'>pip install keras</code>. **Keras** is a high-level deep learning library that sits on top of Tensorflow, and makes it significantly easier to write your own neural networks in just a couple lines of code.
11. Before we start writing some code, let's restart our `conda` environment so that we can be sure that all the installations are complete: <code class='language-bash'>source deactivate caispp</code> (to deactivate our environment), and then <code class='language-bash'>source activate caispp</code> (to reactivate it).


<p>
  As an ending note, we will not spend too much time covering the basics of Python (e.g. syntax). If you feel like you need to brush up on your Python, go <a href='https://docs.python.org/3/tutorial/'>here</a>. We also will assume some knowledge of Calculus and Linear Algebra (mostly just matrix multiplication). If you're feeling kind of iffy on these topics, try to find some time to read through chapters 2 and 3 of <a href='http://www.deeplearningbook.org/'>this online deep learning book</a>.
</p>

<p>
  That's it for now! If you have any questions, or if any of these instructions do not work, please email us at caisplus@usc.edu. Chances are some of you will run into the same problems, so we want to update this lesson with any trouble shooting advice we can get.
</p>
