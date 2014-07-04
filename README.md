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
