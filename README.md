# FactoringWithSympyNullspace
Python Factorization using Sympy Nullspace and Matrix functions. 
This only works with numbers with two factors right now, i need
to fix that but i wanted to at least publish to github. There is
a bug when there are more than two factors you will only get a couple
of factors correct but i will fix this soon. 

This works really well but gauss elimination is slowing it down. I have
LU Decomposition code in here too and that should really speed it up
but it's not working correctly just yet. 

```
Factorization using sympy Matrix and Null space operations.
from fwsm import factorise
Here are some samples for usage i don't automatically adjust B and I
so you may have to play with the numbers for effeciency

In [91]: factorise(1009732533765211, 2500, 10000)
Found 20 potential solutions
11344301 89007911
Out[91]: [mpz(11344301), mpz(89007911)]

In [93]: factorise(32990125356016687985769067)
Found 46 potential solutions
4898499751721 6734740640627
Out[93]: [mpz(4898499751721), mpz(6734740640627)]

This takes a few minutes. Gauss Elimination is slow, hopefully the LU Decomposition
speeds it up.
In [39]: factorise(mpz(603441351914044057309903171375207),  B=35000, I=100000000)
Found 156 potential solutions
28539451776004699 21144111549521893
Out[39]: [mpz(21144111549521893), mpz(28539451776004699)]

```

```
Added LU Decomposition but it's not working, you just have to comment
out the current gauss_elim and uncomment the other two lines if you want
to try to fix it. It would really speed it up.
```
