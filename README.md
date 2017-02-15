# KJIT 

The goal of this project is to learn a kernel-based message operator which
takes as input all incoming messages to a factor and produces a projected
outgoing expectation propagation (EP) message. In ordinary EP, computing an
outgoing message may involve solving a difficult integral for minimizing the KL
divergence between the tilted distribution and the approximate posterior. Such
operator allows one to bypass the computation of the integral by directly
mapping all incoming messages into an outgoing message. Learning of such an
operator is done online during EP.  The operator is termed **KJIT** for
**K**ernel-based **J**ust-**I**n-**T**ime learning for passing EP messages.

Full details are in our [UAI
2015 paper](http://auai.org/uai2015/proceedings/papers/235.pdf).
Supplementary matrial is [here](http://auai.org/uai2015/proceedings/supp/239_supp.pdf).


    Wittawat Jitkrittum, Arthur Gretton, Nicolas Heess, 
    S. M. Ali Eslami, Balaji Lakshminarayanan, Dino Sejdinovic, and Zoltán Szabó
    "Kernel-Based Just-In-Time Learning for Passing Expectation Propagation Messages"
    UAI, 2015


This project extends 

    Nicolas Heess, Daniel Tarlow, and John Winn. 
    “Learning to Pass Expectation Propagation Messages.” 
    NIPS, 2013.
    http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/1493.pdf.

and 

    S. M. Ali Eslami, Daniel Tarlow, Pushmeet Kohli, and John Winn
    "Just-In-Time Learning for Fast and Flexible Inference." 
    NIPS, 2014.
    http://papers.nips.cc/paper/5595-just-in-time-learning-for-fast-and-flexible-inference.pdf


## License
KJIT software is under MIT license.

The KJIT software relies on
[Infer.NET](http://research.microsoft.com/en-us/um/cambridge/projects/infernet/download.aspx)
(freely available for non-commercial use) which is not included in our software. Even
though the license of KJIT software is permissive, Infer.NET's license is not. Please
refer to [its
license](http://research.microsoft.com/en-us/downloads/710cd61f-3587-44f4-b12d-a2c75722c4f6/InferNetLicense.rtf)
for details.

## Repository structure 
The repository contains a number of components.

1. **Poster and paper source files** are in the topmost folders i.e., `dali2015_poster` 
and `uai2015`.
2. **Matlab code** for experimenting in a batch learning setting. Experiments on new
  kernels, factors, random features, message operators are all done in Matlab
in the first stage. Once the methods are developed, they are reimplemented in
C# to be operable in Infer.NET framework. EP inference is implemented in C#
using Infer.NET, not in Matlab. All Matlab code is in the `code` folder.
3. **C# code** for message operators in Infer.NET framework. The code for this
   part is in `code/KernelEP.NET` which contains a C# project developed with
[Monodevelop](http://www.monodevelop.com/) (free cross-platform IDE) on Ubuntu
14.04.  You should be able to use Visual studio in Windows to open the project
file if it is more preferable.

All the code is written in Matlab and C# and expected to be cross-platform.


## Include Infer.NET 
The Matlab part of this project does not depend on the Infer.NET package. 
However, to use our KJIT message operator in the Infer.NET framework, you have to
include Infer.NET package by taking the following steps.

1. Download Infer.NET package from its [Microsoft research
   page](http://research.microsoft.com/en-us/um/cambridge/projects/infernet/).
Upon extracting the zip archive, you will see subfolders including `Bin`, `Source`, 
and its license. Carefully read its license.
2. Copy `Infer.Compiler.dll` and `Infer.Runtime.dll` from the `Bin` folder of 
the extracted archive into `code/KernelEP.NET/lib/Infer.NET/Bin/` of this
repository. Without this step, when you open the project in Monodevelop, it will 
not compile due to the missing dependency.
3. Try to build the project. There should be no errors.


## Useful submodules

In the development of the code for learning an EP message operator, some commonly 
used functions are reimplemented to better suit the need of this project. 
These functions might be useful for other works. These include

* **Incomplete Cholesky factorization**. This is implemented in Matlab in such
  a way that any kernel and any type of data (not necessarily points from
Euclidean space) can be used. The full kernel matrix is not pre-loaded.  Only
one row of the kernel matrix is computed at a time, allowing a large kernel
matrix to be factorized. In this project, points are distributions and the
kernel takes two distributions as input. See `IncompChol`.

* **Dynamic matrix** in Matlab. This is a matrix whose entries are given by a
  function `f: (I, J) -> M` where `I, J` are index list and `M` is a submatrix
specified by `I, J`. The dynamic matrix is useful when the underlying matrix is
too large to fit into memory but entries can be computed on the fly when
needed. In this project, this object is used to represent the data matrix when
a large number of random features are used.  Multiplication (to a regular
matrix or a dynamic matrix) operations are implemented.  See `DynamicMatrix`
and `DefaultDynamicMatrix`.

## Code usage  
Please feel free to contact me (see [wittawat.com](http://wittawat.com))
regarding code usage. For fun, visualization of this repository is available 
[here](https://www.youtube.com/watch?v=m93S5V5zyKw). 


