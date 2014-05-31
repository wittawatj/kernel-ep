kernel-ep
=========

This project is an attempt to learn a kernel-based operator which takes as input all incoming messages to a factor and produces a projected outgoing message. The projected outgoing message is constrained to be a certain parametric form e.g., Gaussian. In ordinary expectation propagation, computing an outgoing message may involve solving a difficult (potentially multdimensional) integral for minimizing the KL divergence between the tilted distribution and the approximate poster. Such operator allows one to bypass the computation of the integral by directly mapping all incoming messages into an outgoing message. Learning of such mapping is done offline with importance sampling used to compute ground truth projected output messages. A learned operator is useful in an application such as tracking where inference has to be done in real time and numerically computing the integral is infeasible due to time constraint. 

