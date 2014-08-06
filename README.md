kernel-ep
=========
This project is an attempt to learn a kernel-based operator which takes as
input all incoming messages to a factor and produces a projected outgoing EP
message. The projected outgoing message is constrained to be a certain
parametric form e.g., Gaussian. In ordinary expectation propagation, computing
an outgoing message may involve solving a difficult (potentially
multdimensional) integral for minimizing the KL divergence between the tilted
distribution and the approximate posterior. Such operator allows one to bypass
the computation of the integral by directly mapping all incoming messages into
an outgoing message. Learning of such mapping is done offline with the aid of
importance sampling for computing ground truth projected output messages. A
learned operator is useful in an application such as tracking where inference
has to be done in real time and numerically computing the integral is
infeasible due to time constraint. 

This project extends the following work

    Heess, Nicolas, Daniel Tarlow, and John Winn. 
    “Learning to Pass Expectation Propagation Messages.” 
    In Advances in Neural Information Processing Systems 26, 
    edited by C. j c Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. q Weinberger, 
    3219–27, 2013. 
    http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/1493.pdf.

Useful Functions
----------------

In the development of the code for learning an EP message operator, some commonly 
used functions are reimplemented to better suit the need of this project. 
These functions might be useful for other researches. These include

* **Incomplete Cholesky factorization**. This is implemented in such a way that 
any kernel and any type of data (does not have to be points from Euclidean space)
can be used. The full kernel matrix is not pre-loaded.
Only one row of the kernel matrix is computed at a time, allowing a large kernel 
matrix to be factorized. 

* **Dynamic matrix**. This is a matrix whose entries are given by a function 
`f: (I, J) -> M` where `I, J` are index list and `M` is a submatrix specified by 
`I, J`. The dynamic matrix is useful when the underlying matrix is very large but 
entries can be computed on the fly when needed. In this project, this object is 
used to represent the data matrix when a large number of random features are used.
Multiplication (to a regular matrix or a dynamic matrix) operations are implemented.

