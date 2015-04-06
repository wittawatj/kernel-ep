# KJIT 

The goal of this project is to learn a kernel-based message operator which
takes as input all incoming messages to a factor and produces a projected
outgoing expectation propagation (EP) message. In ordinary EP, computing an
outgoing message may involve solving a difficult integral for minimizing the KL
divergence between the tilted distribution and the approximate posterior. Such
operator allows one to bypass the computation of the integral by directly
mapping all incoming messages into an outgoing message. Learning of such an
operator is done online during EP.  The operator is termed **KJIT** for
Kernel-based Just-In-Time learning to pass EP messages.

More technical details can be found on [this
page](http://wittawat.com/pages/kernel_ep.html) or in the following paper.

    Wittawat Jitkrittum, Arthur Gretton, Nicolas Heess, S. M. Ali Eslami, Balaji Lakshminarayanan, Dino Sejdinovic, and Zoltán Szabó
    Kernel-Based Just-In-Time Learning for Passing Expectation Propagation Messages
    [arXiv:1503.02551](http://arxiv.org/abs/1503.02551), 2015


This project extends 

    Heess, Nicolas, Daniel Tarlow, and John Winn. 
    “Learning to Pass Expectation Propagation Messages.” 
    In Advances in Neural Information Processing Systems 26, 
    edited by C. j c Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. q Weinberger, 
    3219–27, 2013. 
    http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/1493.pdf.

and 

    Eslami, S. M. A.; Tarlow, D.; Kohli, P. & Winn, 
    "Just-In-Time Learning for Fast and Flexible Inference." 
    In Advances in Neural Information Processing Systems 27, 2014, 154-162
    http://papers.nips.cc/paper/5595-just-in-time-learning-for-fast-and-flexible-inference.pdf


## License
KJIT software is under MIT license.

The KJIT software relies on
[Infer.NET](http://research.microsoft.com/en-us/um/cambridge/projects/infernet/download.aspx)
which is not included in our software. Even though the license of KJIT software
is permissive, Infer.NET is not. Please refer to [its
license](http://research.microsoft.com/en-us/downloads/710cd61f-3587-44f4-b12d-a2c75722c4f6/InferNetLicense.rtf)
for details.

## Repository structure 
The repository contains mainly three components of interest.

* Poster and paper source files are in the top most folders i.e., `dali2015_poster` 
and `uai2015`.
* Matlab code for experimenting in a batch learning setting. Experiments on new
  kernels, factors, random features, message operators are all done in Matlab
in the first stage. Once the methods are developed, they are reimplemented
in C# to be operable in Infer.NET framework. All Matlab code is in the `code`
folder.
* Actual message operators implemented in Infer.NET framework in C#. 
The code for this part is in `code/KernelEP.NET` which contains a C# project 
developed with [Monodevelop](http://www.monodevelop.com/) on Ubuntu 14.04. 
You should be able to use Visual studio in Windows to open the project file if
it is more preferable.

All the code is expected to be cross-platform.


## Include Infer.NET
The Matlab part of this project does not depend on the Infer.NET package. 
However, to use our KJIT message operator in Infer.NET framework, you have to
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


## Useful components

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

## Documentation 
I am well aware that code without any documentation is not useful. 
I will gradually put up documents for the code in Wiki of this github repository.
Please feel free to contact me regarding code usage.

