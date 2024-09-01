# Theory

## Coupon Bonds with stochastic Interest Rates

This report focuses on the valuation of coupon bonds, which provide continuous payouts in the form of coupons and yield a specific amount, known as the face value, at the maturity date. We denote the value of the bond at time $t$, maturing at time $T$, as $B(r,t;T)$, where $r$ represents the stochastic interest rate. The risk-neutral process followed by the interest rate is given by

$$ dr = \kappa(\theta e^{\mu t} - r)dt + \sigma r^{\beta} dW, $$

where the constants are given by $\kappa = 0.08116$, $\theta=0.0409$, $\mu = -0.0222$, $\sigma = 0.251$, $\beta = 0.527$. The market value of the coupon bond $B(r,t;T)$ satisfies the following PDE

$$ \frac{\partial B}{\partial t} + \frac{1}{2} \sigma^2 r^{2\beta} \frac{\partial^2 B}{\partial r^2} + \kappa(\theta e^{\mu t} - r) \frac{\partial B}{\partial r} - rB + Ce^{-\alpha t} = 0, $$

\noindent
if the bond pays out continuous coupons at the rate

$$Ce^{-\alpha t}$$

for the constants $C$ and $\alpha$ that are specified by the bond contract in this report to be $C=1.07$ and $\alpha = 0.01$. The domain of the problem is $r \in [0, \infty)$ and $t < T$. The Boundary conditions on $B(r,t;T)$ are given by

$$B(r, t = T; T) = F;$$



$$\frac{\partial B}{\partial t} + \kappa\theta e^{\mu t} \frac{\partial B}{\partial r} + Ce^{-\alpha t} = 0 \text{ at } r = 0;$$



$$B(r, t; T) \to 0 \text{ as } r \to \infty.$$\end{equation}

The last condition is known as the Dirichlet condition, and it can be replaced by the weaker Neumann condition

$$ \frac{\partial B}{\partial r} \to 0 \text{ as } r \to \infty$$

## Options on a Bond

If we consider a put option $V$, then we can denote the value of the option to sell the coupon bond $B(r,t;T)$ at time $T_1$ by $V(r,t;T_1,T)$. It can be shown that, on the domain $r \in [0, \infty)$ and $t < T_1$, that the option value $V(r,t;T_1,T)$ follows the PDE

$$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 r^{2\beta} \frac{\partial^2 V}{\partial r^2} + \kappa(\theta e^{\mu t} - r) \frac{\partial V}{\partial r} - rV = 0.$$

For an American put option that expires at $t=T_1$ we have

$$V(r,t=T_1;T_1,T) = \max(X-B(r,T_1;T),0)$$

and if the option to sell the bond is exercised early

$$V(r,t;T_1,T) = X - B(r,t;T)$$

According to the no-arbitrage principle and early exercise, the option must satisfy the boundary conditions

$$V(r, t; T_1, T) \geq \max(X - B(r, t; T), 0), \text{ for } t \leq T_1;$$

$$\frac{\partial V}{\partial t} + \kappa\theta e^{\mu t} \frac{\partial V}{\partial r} = 0 \text{ at } r = 0; $

$$V(r, t; T_1, T) \to X - B(r,t;T) \text{ as } r \to \infty.$$

If we instead have a European put option then

$$V(r, t=T_1; T_1, T) = \max(X - B(r, t; T), 0)$$

$$\frac{\partial V}{\partial t} + \kappa\theta e^{\mu t} \frac{\partial V}{\partial r} = 0 \text{ at } r = 0;$$

$$V(r,t;T_1,T) \to 0 \text{ as } r \to \infty$$

or 

$$\frac{\partial V}{\partial r} \to 0 \text{ as } r \to \infty$$

# The Crank-Nicolson method

We can use finite difference methods in the form of the Crank-Nicolson scheme to approximate the value of derivatives within our problem. The Crank-Nicolson scheme works by evaluating the derivatives at $V(S,t + \Delta t/2)$, resulting in a theoretical error in the time derivative of $(\Delta t)^2$. If we have some function $B(r, t)$ over a domain $r \in [0,r_{max}]$ and $t \in [0, T]$, then we can discretise the domains by defining $j_{max}$ subintervals of length $\Delta r = r_{max}/j_{max}$ and $i_{max}$ subintervals of length $\Delta t = T/i_{max}$. Therefore the nodes of the grid are given by $0, \Delta r, 2 \Delta r,..., j_{max} \Delta r$ and $0, \Delta t,..., i_{max} \Delta t$. The value of $B(r,t)$ at each node in the grid is then given by $B(j \Delta r, i \Delta t) = B_j^i$. 
