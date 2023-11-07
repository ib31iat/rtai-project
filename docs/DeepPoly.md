# DeepPoly.py

## Matrix form of backsubstitution
Let $x_{i+1}, x_{i}$ and $x_{i-1}$ be the outputs of layers $i+1, i$ and $i-1$ respectively. Assume we have the linear bounds 
$$x_{i+1}\leq Wx_{i}+b \ \ \ \ \text{and} \ \ \ \ Lx_{i-1}+l\leq x_{i}\leq Ux_{i-1}+u.$$
Then, we have
$$\begin{align*}
x_{i+1,k} &\leq \sum_m W_{km}x_{i,m}+b_k \\
&\leq \sum_{m:\ W_{km}\geq0} W_{km}\left(\sum_n U_{mn}x_{i-1,n}+u_m\right) + \sum_{m:\ W_{km}<0} W_{km}\left(\sum_n L_{mn}x_{i-1,n}+l_m \right) + b_k\\
&=\sum_n\left(\sum_{m:\ W_{km}\geq0}W_{km}U_{mn}+\sum_{m:\ W_{km}<0}W_{km}L_{mn}\right)x_{i-1,n}+\sum_{m:\ W_{km}\geq0}W_{km}l_{m}+\sum_{m:\ W_{km}<0}W_{km}l_{m}+b_k.
\end{align*}
$$
Hence, we have 
$$
x_{i+1}\leq \Big(\max(0, W)U+\min(0,W)L\Big)x_{i-1}+\max(0, W)u+\min(0,W)l+b.
$$
We can derive the formula for the lower bound analogously.
## Leaky ReLU
Let $\sigma$ denote the slope of a leaky ReLU on the negative axis. Assume the inputs to the leaky ReLU are contained in an interval $[l,u]$ with $l<0<u$. 
#### Case 1: $\ \sigma\leq 1$
If $\sigma\leq1$, there is a unique tightest linear upper bound in terms of the input $x$ on the outputs $y$ given by 
$$
y\leq \frac{u-\sigma l}{u-l}x+\frac{(\sigma-1)ul}{u-l}.
$$
There are many possible linear lower bounds. They are given by
$$
y\geq \alpha x, \ \ \  \alpha\in[\sigma, 1].
$$
Note that the vanilla ReLU belongs to this case as ReLU is the same as leaky ReLU with $\sigma=0$.

#### Case 2: $\ \sigma\geq1$
In this case, there is a unique tightest lower bound, given by
$$
y\geq \frac{u-\sigma l}{u-l}x+\frac{(\sigma-1)ul}{u-l}.
$$
There are many possible linear upper bounds. They are given by
$$
y\leq \alpha x, \ \ \  \alpha\in[1, \sigma].
$$