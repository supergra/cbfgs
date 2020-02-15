This is a BFGS optimizer.


Features:
- Advanced line search with "zoom" and many safeguards, following Nocedal
  and Wright, especially Chapters 3 and 7.
- New techniques for solving optimization problems with significantly more
  noise in the function evaluation than in the gradient. This could be
  relevant for numerical applications where an accurate discrete gradient
  can be computed around a given partially converged solution.

`make <clean, new, install>`

`./cbfgs`

To select different coded examples, edit main.c

To change optimizer settings, edit bfgs.c

To experiment with noisy functions, set CONVEX_NOISY = 1 in bfgs.c and
use the noisy function in main.c


