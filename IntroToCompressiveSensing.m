%% INTRODUCTION TO COMPRESSIVE SENSING
% This script is an introduction to sparse signal representation and
% compressive sensing.
%%
close all
clearvars
clc

%% Basis Representation Fundamentals
% Every signal $x\in\mathbf{R}^N$ is representable in terms of $N$ coefficients
% $\{s_i\}_{i=1}^N$ in a given basis $\{\psi_i\}_{i=1}^{N}$ for $\mathbf{R}^N$ as 
%
% $$x= \sum\limits_{i=1}^N \psi_i s_i$$
%
% Arranging the $\psi_i$ as columns into the $N\times N$ matrix $\Psi$ and the
% coefficients $s_i$ into the $N\times 1$ coefficient vector $\textbf{s}$, we can
% write that $x=\Psi s$, with $s\in\mathbf{R}$.
%
% We say that signal $x$ is $K$-sparse in the basis $\Psi$ if there exists
% a vector $s\in\mathbf{R}^N$ with only $K\ll N$ nonzero entries such that $x=\Psi s$.
%
% If we use basis matrix containing $N$ unit-norm column vectors of length
% $L$ where $L<N$ (i.e. $\Psi\in\mathbf{R}^{L\times N}$), then for any
% vector $x\in\mathbf{R}^L$ there exist infinitely many decompositions
% $s\in\mathbf{R}^N$ such that $x=\Psi s$. We refer to $\Psi$ as the
% overcomplete sparsifying dictionary.
%
% Sparsifying dictionary can be:
%
% # chosen from a known set of transforms like DCT, wavelet...
% # learnt from data 
%
%
%%% Example dictionary for 2D-DCT transformation
figure
visualizeDictionary(kron(dctmtx(8),dctmtx(8))')
title('2D-DCT Dictionary')
figure
haar1D = full(wmpdictionary(8, 'lstcpt', {{'haar',3}}));
visualizeDictionary(kron(haar1D, haar1D))
title('2D-DWT Dictionary - HAAR')

%% Uncertainty Principle
% As an illustrative example, we will consider the case where our
% dictionary is the union of two particular orthobases: the identity
% (spike) basis and the Fourier (sine) basis $\Psi = [I\quad F]$.
%
% Spike and Fourier basis are mutually fully incoherent in the sense that
% it takes $n$ spikes to build up a single sinusoid and also it takes $n$
% sinusoids to build up a single spike.
%


n = 64;

fourier = (1/sqrt(n))*exp((1j*2*pi*[0:n-1]'*[0:n-1])/n);
spike = eye(n);

psi = [spike, fourier];

% figure
imagesc(real(psi))
axis image
title('Spike/Fourier Dictionary')

%% Sines and spikes example
% Now we will create a signal that is a mixture of spikes and sines. As we
% know that the first half of our matrix $\Psi$ contains spike functions
% and the second half corresponds to sine functions, we can construct
% random sparsity pattern with sparsity K and compute $x=\Psi*s$ to obtain
% a signal which is a mixture of impulses and sinusoids.

%desired sparsity K
K = 2;
%randomly selected basis coefficients
idx = randi([1,2*n], 1, K);

%random sparsity vector 
s = zeros(2*n, 1);
s(idx) = 1;

%obtain the signal which is a mixture of impulses and sinusoids
x = psi*s;

%visualize signal
% figure
stem(real(x))
title('Signal - mixture of K impulses and sinusoids')


%% Minimum Energy Decomposition - $l_2$ norm
% There are infinite number of ways we can decompose signal x using atoms
% from our dictionary. Most natural way would be to use certain basis
% functions which correspond to previously selected basis function indices.
% This way we get the sparsest possible representation.
%
% Another way we can get representation for x is by applying $\Psi^*$ 
% and by dividing the result by 2. Since $\Psi\Psi^*=2I$, 
% we get next reproducing formula for x.
%
% $$x=\frac{1}{2}\Psi(\Psi^*x)$$
%
% When we apply $s=\frac{1}{2}\Psi^*x$ we get result that corresponds to
% the minimum energy decomposition of our signal into a coefficient vector
% that represents x. Minimum energy solution corresponds to $l_2$ norm
% minimizaton. Unfortunately, minimum energy decomposition almost never
% yields the sparsest possible soultion. The reason for this is that the
% vector that a vector has minimum nenergy when its total energy is
% distribured over all the coefficients of the vector. $l_2$ gives us a
% solution that is dense, but has small values for the coefficients.
%
% Our ability to separate sines part from the spikes part of our signal of
% interest is what will determine whether or nor we can find a unique
% sparse decomposition. Being able to tell these two components apart comes
% from a new kind of uncertainty principle which states that a signal
% can't be too sparse in time and frequency domain simoultaneously.


%minimum energy decomposition
s = psi'*x;

% figure
stem(real(s))
title('Minimum energy decomposition - l_2 norm')

%% Sparse decomposition
% Since our goal of finding sparsest possible representation of our signal
% $x$ over some basis $\Psi$ is equivalent to finding the solution with the
% smallest number of nonzero elements in basis coefficient vector $s$ we
% will use $l_0$ pseudo-norm to find our solution.
% Sparse signal recovery can be formulated as finding minimum-cardinality
% solution to a constrained optimization problem. In the noiseless case,
% our constraint is simply $x=\Psi s$, while in the noisy case(assuming
% Gaussian noise), the solution must satisfy $\Vert
% x-x^*\Vert_2\leq\epsilon$ where $x^*=\Psi s$ is the hypothetical
% noiseless representation and the actual representation is
% $\epsilon$-close to it in $l_2$ norm. The objective function is the
% cardinality of s(number of nonzeros) which is often denoted $\Vert
% x\Vert_0$ and called $l_0$ norm of $s$. 
%
% Optimization problems corresponding to noiseless and noisy sparse signal
% recovery can be written as:
%
% * $\min\limits_x\Vert x\Vert_0 \quad s.t. \quad x=\Psi s$
% * $\min\limits_x\Vert x\Vert_0 \quad s.t. \quad \Vert x-\Psi s\Vert_2\leq\epsilon$
%
% In general, finding a minimum-cardinality solution satisfying linear
% constraints is an NP-combinatorial problem and an approximation is
% necessary to achieve computational efficiency. Two main approximation
% approaches are typically used in sparse recovery: the first one is to
% address the original NP-combinatorial problem via approximate methods,
% such as greedy search, while the second one is to replace the intractable
% problem with its convex relaxation that is easy to solve. In other words,
% one can either solve the exact problem approximately, or solve an
% approximate problem exactly.
%
% In figure below, we can see sparse decomposition of our signal of
% interest. Notice that there are only K coefficients active, while others
% are equal to zero and that is exactly what we wanted to achieve.

%sparse decomposition
nIter = 10;
s = sparseCode(x, psi, K, nIter);

% figure
stem(real(s))
title('Sparse decomposition - l_0 norm')

%% Sparse Recovery Problem Formulations
% We will use the following notation in this section: $x=(x_1,...,X_N)\in
% \mathbf{R}N$ is an unobserbed sparse signal, $y=(y_1,...,y_M)\in
% \mathbf{R}^M$ is a vector of measurements(observations), and
% $A=\{a_{ij}\}\in\mathbf{R}^{M\times N}$ is a design matrix.
%
% The simplest problem we are going to start with is the noiseless signal
% recovery from a set of linear measurements, i.e. solving for x the system
% of linear equations:
%
% $$y=Ax$$
%
% It is usually assumed that $A$ is a full-rank matrix, and thus for any $y\in\mathbf{R}^M$,
% the above system of linear equations has a solution. Note that when the number of unknown
% variables, i.e. dimensionality of the signal, exceeds the number of observations, i.e.
% when %N\geq M%, the above system is underdetermined, and can have 
% infinitely many solutions. In order to recover the signal $x$, 
% it is necessary to further constrain, or regularize, the problem. 
% This is usually done by introducing an objective function, 
% or regularizer R(x), that encodes additional properties of the signal, 
% with lower values corresponding to more desirable solutions. 
% Signal recovery is then formulated as a constrained optimization problem:
%
% $$\min\limits_{x\in\mathbf{R}^N} R(x)\quad s.t.\quad y=Ax$$
%
% In general, $l_q$ norms for particular values of $q$, denoted $\Vert
% x\Vert_q$ or more precisely their $q$-th power $\Vert x\Vert_q^q$ are
% frequently used as regularizers $R(x)$ in constrained optimization
% problems.
%
% For a $q\ge 1$, the $l_q$ norm, also called just $q$-norm of a vector
% $x\in\mathbf{R}^N$ is defined as:
%
% $$\Vert x\Vert_q=(\sum\limits_{i=1}^N\vert x_i\vert^q)^\frac{1}{q}$$
%
% We can now observe relation between cardinality and $\Vert l_q\Vert$-norms. 
% The function $\Vert x\Vert_0$ referred to as $l_0$-norm of $x$ 
% is defined as a limit of $\Vert x\Vert_q^q$ when $q\to0$:
% 
% $$\Vert x\Vert_0=\lim\limits_{q\to 0}\Vert x\Vert_q^q=\lim\limits_{q\to 0}\sum\limits_{i=1}^p\vert x_i\vert^q=\sum\limits_{i=1}^p\lim\limits_{q\to 0}\vert x_i\vert^q$$
%
% For each $x_i$, when $q\to 0$, $\vert x_i\vert^q\to I(x_i)$, the
% indicator function, which is 0 at $x=0$ and 1 otherwise.
% Thus, $\Vert x\Vert_0=\sum\limits{i=1}^p I(x_i)$, which gives exactly the
% number of nonzero elements of vector x called cardinality.
% Using the cardinality function, we can now write the problem of sparse
% signal recovery from noiseless linear measurements as:
%
% $$\min\limits_x\Vert x\Vert_0 \quad s.t. \quad y=A x$$
%
% As already mentioned before, the above optimization problem is NP-hard
% and no known algorithm can solve it efficiently in polynomial time.
% Therefore, approximations are necessary and were already presented
% before. Under appropriate conditions the optimal solution can be
% recovered efficiently by certain appproximate techniques.
% 
% First approach to approximation is a heuristic-based search such as
% gready search. In gready search method, one can start with a zero vector
% and keep adding nonzero coefficients one by one, selecting at each step
% the coefficient that leads to the best improvement in the objective
% function(gready coordinate descent). In general, such heuristic search
% methods are not guaranteed to find the global optimum. However, in
% practive, they are simple to implement, very computationally efficient
% and under certain conditions they are even guaranteed to recovel the
% optimal solution.
%
% 
% An alternative approximation technique is the relaxation approach based on replacing 
% an intractable objective function or constraint by a tractable one.  
% For example, convex relaxations approximates a non-convex optimization problem 
% by a convex one, i.e. by a problem with convex objective and convex constraints. 
% Such problems are known to be "easy", i.e. there  exists a variety of efficient
% optimization methods for solving convex problems. 
% Clearly, besides being easy to solve, e.g., convex, the relaxed version
% of the optimization problem must also enforce solution sparsity. 
% In the following sections, we discuss lq-norm based relaxations,
% and show that the Zi-norm occupies a unique position among them,
% combining convexity with sparsity.
% A convex optimization problem is minimization of a convex function over 
% a convex set of feasible solutions defined by the constraints.
% Convex problems are easier to solve than general optimization problems 
% because of the important property that any local minima 
% of a convex function is also a global one. 
%
%% Relaxations of Original Sparse Recovery Problem
% We will focus on the $l_q$-norms as possible relaxations of $l_0$-norm.
% These functions are convex for $q\geq 1$ and nonconvex for $q<1$.
% For example, $l_2$-norm(Euclidean norm) is natural first choice as a
% relaxation of $l_0$-norm. Our sparse recovery problem now writes:
%
% $$\min\limits_x\Vert x\Vert_2^2 \quad s.t. \quad y=A x$$
%
% Using $l_2$-norm as an objective has several advantages some of which are
% its convexity and thus its property that it has a unique minimum, and
% finally it solution is available in a closed form. The closed form
% solution to this problem is also known as pseudo-inverse solution of
% $y=Ax$ when $A$ has more columns than rows(we assume that $A$ is full-rank, 
% i.e. all of its rows are linearly independent).
%
% However, despite its convenient properties, $l_2$-norm has a serious
% drawback when it comes to sparse recovery. It turns out that the optimal
% solution obtained by pseudo-inverse is practically never sparse.
%
% To understand why the l2-norm does not promote the solution sparsity while the
% $l_1$-norm does, and to understand the convexity and sparsity-inducing
% properties of $l_q$-norms in general, let us consider the geometry of a 
% sparse recovery problem, where $\Vert x\Vert_q^q$ replaces the 
% original cardinality objective $\Vert x\Vert_0$ :
%
% $$\min\limits_x\Vert x\Vert_q^q \quad s.t. \quad y=A x$$
%
% Sets of vectors with same value of the function $f(x)$,i.e. $f(x)=const$,
% are called the level sets of $f(x)$. For example, the level sets of $\Vert x\Vert_q^q$
% function are vector sets with same $l_q$-norm. A set of vectors satisfying 
% $\Vert x\Vert_q^q \leq r^q$ is called an $l_q$-ball of radius r;
% its “surface” (set boundary) is the corresponding level set $\Vert x\Vert_q^q = r^q$.
% Note that the corresponding $l_q$-balls bounded by the level sets are convex for $q\geq 1$ 
% (line segments between a pair of its points belong to the ball), and nonconvex for
% $0<q<1$ (line segments between a pair of its points do not always belong to the ball).
% 
% <<lq_balls.png>>
% 
% From a geometric point of view, solving the optimization problem(Pq) is 
% equivalent to “blowing up” $l_q$-balls with the center at the origin, 
% i.e., increasing their radius, starting from 0, until they touch the hyperplane Ax = y.
% The resulting point is the minimum $l_q$-norm vector that is also a feasible point, 
% i.e. it is the optimal solution of sparse recovery problem.
% 
% <<lq_balls_solution.png>>
% 
%
% Note that when $q\leq 1$, $l_q$-balls have sharp “corners” on the coordinate axis,
% corresponding to sparse vectors, since some of their coordinates are zero, 
% but $l_q$-balls for $q>1$ do not have this property. 
% Thus, for $q \leq 1$, $l_q$-balls are likely to meet the hyperplane $Ax = y$
% at the corners, thus producing sparse solutions, while for $q>1$ the 
% intersection practically never occurs at the axes, and thus solutions are not sparse.
%
% Within the family of $\Vert x\Vert_q^q$ functions, only those with $q\geq 1$ are
% convex, but only those with $0<q\leq 1$ are sparsity-enforcing. 
% The only function within that family that has both useful properties 
% is therefore $\Vert x\Vert_1$, i.e.the $l_1$-norm.
% This unique combination of sparsity and convexity is the reason 
% for the widespread use of $l_1$-norms in the modern sparse signal
% recovery field.
% Optimization problem using $l_1$ norm writes:
%
% $$\min\limits_x\Vert x\Vert_1 \quad s.t. \quad y=A x$$
%

x = [-1:0.01:1];

figure
hold on
plot(x, (sum(abs(x).^2, 1)))
plot(x, (sum(abs(x).^1, 1)))
plot(x, (sum(abs(x).^(1/2), 1)))
plot(x, (sum(abs(x).^(1/100), 1)))
plot(x, (sum(abs(x).^(realmin), 1)))
axis tight

%% Compressed Sensing
% The key idea behind compressive sensing is that the majority of real-life
% signals(images, audio...) can be well approximated by sparse
% representation vectors, given some appropriate basis $\Psi$, and that
% exploiting the sparse signal structure can dramatically reduce the signal
% acquisition cost. Traditional approach to signal acquisition is based on the classical Shannon-
% Nyquist result stating that in order to preserve information about a signal, 
% one must sample the signal at a rate which is at least twice the signal's bandwidth, 
% defined as the highest frequency in the signal's spectrum. Note, however,
% that such classical scenario gives a worst-case bound, since it does not take
% advantage of any specific structure that the signal may possess. In practice,
% sampling at the Nyquist rate usually produces a tremendous number of samples,
% e.g., in digital and video cameras, and must be followed by a compression step
% in order to store or transmit this information efficiently.
%
% The compression step uses some basis to represent a signal 
% (e.g., Fourier, wavelets, etc.) and essentially throws away a large fraction
% of coefficients, leaving a relatively few important ones.
% Thus, a natural question is whether the compression step can
% be combined with the acquisition step, in order to avoid the collection
% of an unnecessarily large number of samples.
%
% Compressive sensing offers positive answer to the above question. Let
% $x\in\mathbf{R}^N$ be a signal that can be represented sparsely in some
% basis $\Psi$ i.e. $x=\Psi s$ where $\Psi$ is an $N\times N$ matrix of
% basis vectors(columns), and where $s\in\mathbf{R}^N$ is a sparse vector
% of the signal's coordinates with only $K\ll N$ nonzeros. Though the
% signal is not observed directly, we can obtain a set of linear
% measurements:
%
% $$y=\Phi x=\Phi\Psi s=A s$$
%
% where $\Psi$ is an $N\times M$ measurement matrix and $y\in\mathbf{R}^M$
% is a set of $M$ measurements or samples where $M$ can be much smaller
% than the original dimensionality of the signal, hence the name
% compressive sensing(CS).
%
% The central problem of compressed sensing is reconstruction of a 
% high-dimensional sparse signal representation $x$ from a low-dimensional
% linear observation $y$.
%%
% 
% <<cs_matrix_form.png>>
% 
%
%% Uniqueness of Sparse Recovery Problem Solution
% In this section we will discuss when the solutions of the $l_0$- and
% $l_1$- norm minimization problems are unique. The main design criteria
% for matrix $A$ is to enable the unique identification of a signal of
% interest $x$ from its measurements $y=Ax$. Clearly, when we consider the
% class of K-sparse signals $\Sigma_K$, the number of measurements $M>K$ for any
% matrix design, since the identification problem has $K$ unknowns.
%
% We will now determine properties of $A$ that guarantee that distinct
% signals $x,x'\in \Sigma_K, x\neq x'$, lead to different measurement
% vectors $Ax\neq Ax'$. In other words, we want each vector $y\in
% \mathbf{R}^M$ to be matched to at most one vector $x\in \Sigma_K$ such
% that $y=Ax$.
%
% A key relevant property of the matrix in this context is its spark.
% Given an $M\times N$ matrix $A$, its spark $spark(A)$, is defined as the
% minimal number of linearly dependent columns. Spark is closely related to
% the Kruskal's rank $krank(A)$ defined as the maximal number $k$ such that
% every subset of $k$ columns of the matrix $A$ is linearly independent.
%
% $$spark(A)=krank(A)+1\quad and \quad rank(A)\geq krank(A)$$
%
% By definition, the vectors in the null-space of the matrix $Ax=0$ must
% satisfy $\Vert x\Vert_0\geq spark(A)$, since these vectors combine
% linearly columns from $A$ to give the zero vector, and at least $spark$
% such columns are necessary by definition.
%
% Sparse recovery solution uniqueness via spark can be stated as:
%
% * A vector $\bar{x}$ is the unique solution of the sparse recovery
% problem if and only if $\bar{x}$ is a solution of $Ax=y$ 
% and $\Vert x\Vert_0<\frac{1}{2} spark(A)$
%
% or in alternative formulation:
%
% * If $spark(A)>2K$, then for each measurement vector $y\in\mathbf{R}^M$
% there exists at most one signal $x\in\Sigma_K$ such that $y=Ax$.
%
% We can provide proof for the above theorem. Consider an alternative
% solution $x$ that satisfies the same linear system $Ay=x$. This implies
% that $(\bar{x}-x)$ must be in the null-space of $A$, i.e.
% $A(\bar{x}-x)=0$(the columns of $A$ corresponding to nonzero entries of
% the vector $(\bar{x}-x)$ are linearly dependent). Thus, the number of
% such columns must be greater or equal to $spark(A)$ by definition of
% spark. Since the support of $(\bar{x}-x)$ is a union of supports of
% $\bar{x}$ and $x$, we get $\Vert (\bar{x}-x)\Vert_0 \leq
% \Vert\bar{x}\Vert_0+\Vert x\Vert_0$. But since
% $\Vert\bar{x}\Vert_0<\frac{1}{2} spark(A)$, we get:
%
% $$\Vert x\Vert_0\geq \Vert\bar{x}-x\Vert_0-\Vert\bar{x}\Vert_0>\frac{1}{2} spark(A)$$
%
% which proves that $\bar{x}$ is indeed the sparsest solution.
%
% The singleton bound yields that the highest spark of an matrix
% $A\in\mathbf{R}^{M\times N}$ with $M<N$ is less than or equal to $M+1$
% and using the before stated theorems we get the requirement $M\geq 2K$.
%
% While spark is useful notion for proving the exact recovery of a sparse
% optimization problem, it is NP-hard to compute since one must verify that
% all sets of columns of a certain size are linearly independent. Thus, it
% is preferable to use properties of $A$ which are easily computable to
% provide recovery guarantees.
%
% The coherence $\mu (A)$ of a matrix is the largest absolute inner product
% between any two columns of $A$:
%
% $$\mu (A)=\max\limits_{1\leq i\neq j\leq N}\frac{\langle a_i,
% a_j\rangle}{\Vert a_i\Vert_2 \Vert a_j\Vert_2}$$
%
% For any matrix $A$,
%
% $$spark(A)\geq 1+\frac{1}{\mu (A)}$$
%
% Quite simple way to read the coherence is from the absolute value Gram
% matrix. Gram matrix is defined as $G=A'A$ where we are considering
% conjugate transpose of the matrix A. To read the coherence from Gram
% matrix, we reject the diagonal elements since they correspond to the
% inner product of an atom with itself(for a properly normalized dictionary
% they should be 1 anyway). Since G is symmetric we need to look only upper
% triangular half of it to read off the coherence. The value of coherence
% $\mu (A)$ is equal to largest value in upper triangular part of matrix
% $A$ with diagonal excluded.
%
% It can be shown that $\mu (A)\in [\sqrt{\frac{N-M}{M(N-1)}}, 1]$. The
% lower bound is known as the Welch bound. Note that when $N>>M$, the lower
% bound is approximately $\mu(A)\geq \frac{1}{\sqrt(M)}$
%
% We can show computation of coherence using Gram matrix on example of
% dictionary composed of spike and sine basis functions.

%%
n = 64;

fourier = (1/sqrt(n))*exp((1j*2*pi*[0:n-1]'*[0:n-1])/n);
spike = eye(n);

psi = [spike, fourier];

G = psi'*psi;

coherence = max(max(abs(triu(G,1))))

figure
imagesc(real(G))
title('Gram matrix')


%% Restricted Isometry Property - RIP
% The prior properties of the CS design matrix provide guarantees of
% uniqueness when the measurement vector $y$ is obtained without error.
% There can be two sources of error in measurements: inaccuracies due to
% noise at sensing stage(in the form of additive noise $y=Ax+noise$) and
% inaccuracies due to mismatches between the design matrix used during
% recovery and that implemented during acquisition(in the form of
% multiplicative noise $A'=A+A_noise$). Under these sources of error, it is
% no longer possible to guarantee uniqueness, but it is desirable for the
% measurement process to be tolerant to both types of error. To be more
% formal, we would like the distance between the measurement vectors for
% two sparse signals $y=Ax$ and $y'=Ax'$ to be proportional to the distance
% between the original signal vectors $x$ and $x'$. Such a property allows
% us to guarantee that for small enough noise, two sparse vectors that are far
% appart from each other cannot lead to the same noisy measurement vector.
% This behaviour has been formalized into the restricted isometry
% property(RIP):
%
% * A matrix A has the $(K,\delta)$-restricted isometry
% property($(K,\delta)$-RIP) if, for all $x\in\Sigma_K$,
%
% $$(1-\delta)\Vert x\Vert_2^2\leq \Vert Ax\Vert_2^2\leq (1+\delta)\Vert x\Vert_2^2$$
%
% In words, the (K,\delta)-RIP ensures that all submatrices of $A$ of size
% $M\times K$ are close to an isometry, and therefore distance-preserving.
% This property suffices to prove that the recovery is stable to presence
% of additive noise and the RIP also leads to stability with respect to the
% multiplicative noise introduced by the CS matrix mismatcs $A_noise$.
%
%% Algorithms for Sparse Recovery
% We will focus on the noisy sparse recovery problems:
%
% * $l_0$-norm minimization: $\min\limits_x\Vert x\Vert_0 \quad s.t. \quad \Vert x-\Psi s\Vert_2\leq\epsilon$
% * $l_1$-norm relaxation:   $\min\limits_x\Vert x\Vert_1 \quad s.t. \quad \Vert x-\Psi s\Vert_2\leq\epsilon$
% * Lagrangian form $l_1$ minimization (LASSO): $\min\limits_{x}\frac{1}{2}\Vert y-Ax\Vert_2^2+\lambda\Vert x\Vert_1$
%
% Recall that x is an N-dimensional unknown sparse signal, 
% which in a statistical setting corresponds to a vector of coefficients 
% of a linear regression model, where each coefficient $x_i$ signifies 
% the amount of influence the $i$-th input, or predictor variable $A_i$, 
% has on the output $y$, an $M$-dimensional vector of observations of 
% a target variable $Y$. $A$ is an $M\times N$ design matrix, where the $i$-th 
% column is an $M$-dimensional sample of a random variable $A_i$, i.e. 
% a set of $M$ independent and identically distributed, or i.i.d., observations.
%
% We would like to focus on the specific case of orthogonal design
% matrices. It turns out that in such case both $l_0$- and $l_1$-norm
% optimization problems decompose into independent univariate problems, and
% their optimal solutions can be easily found by very simple univariate
% thresholding procedures.
%
%% Univariate Thresholding
% An orthonormal(orthogonal) matrix $A$ is an $N\times N$ square matrix satisfying
% $A^TA=AA^T=I$ where $I$ denotes the identity matrix. A linear
% transformation defined by an orthogonal matrix $A$ has a nice property,
% it preserves the $l_2$-norm of a vector.
%
% $$\Vert Ax\Vert_2^2=(Ax)^T (Ax) = x^T (A^TA) x = x^T x =\Vert x\Vert_2^2$$
%
% The same is also true for $A^T$ and we get:
%
% $$\Vert y-Ax\Vert_2^2=\Vert A^T(y-Ax)\Vert_2^2=\Vert\hat{x}-x\Vert_2^2=\sum\limits_{i=1}^N(\hat{x}_i-x_i)^2$$
%
% where $\hat{x}=A^Ty$ corresponds to the ordinary least squares(OLS)
% solution when $A$ is orthogonal, i.e. $\hat{x}=\min\limits_{x}\Vert y-Ax\Vert^2$
% This transformation of the sum-squared loss will greatly simplify our
% optimization problems.
%
%% $l_0$-norm Minimization
% The problem of $l_0$-norm minimization can now be rewritten as:
%
% $$\min\limits_x\Vert x\Vert_0 \quad s.t. \quad \sum\limits_{i=1}^N(\hat{x}_i-x_i)^2\le\epsilon^2$$
%
% In other words, we are looking for the sparsest (i.e., smallest $l_0$-norm)
% solution $x^*$ that is $\epsilon$-close in $l_2$-sense to the OLS solution $x = A^T y$. 
%
% It is easy to construct such solution by choosing k largest 
% (in the absolute value) coordinates of x and by setting the rest of the 
% coordinates to zero, where k is the smallest number of such coordinates 
% needed to get $\epsilon$-close to $\hat{x}$, i.e. to make the solution feasible.
%
% This can also be viewed as an univariate hard thresholding of the OLS
% solution $\hat{x}$, namely:
% 
% <<hard_thresholding.png>>
% 
% where $t(\epsilon)$ is a threshold value below the $k$-th largest, but
% above the $(k+1)$-th largest value among $\{\hat{x}_i\}$.

%hard thresholding
hardThresholding = @(x, th) x.*(abs(x)>=th);

x = -1:0.01:1;
threshold = 0.35;
x_ht = hardThresholding(x, threshold);

figure, axis tight
plot(x, x_ht)
hold on
plot([-threshold:0.01:threshold], [-threshold:0.01:threshold], '-.')
title('Hard Thresholding Operator')
xlabel('x')
ylabel('x*')


%% $l_1$-norm Minimization
% For an orthogonal $A$, the LASSO problem becomes:
%
% $$\min\limits_{x}\frac{1}{2}\sum\limits_{i=1}^N(\hat{x}_i-x_i)^2+\lambda \sum\limits_{i=1}^N\vert x_i\vert$$
%
% which trivially decomposes into $N$ independent, univariate optimization
% problems, one per each $x_i$ variable, $i=1,...N$:
%
% $$\min\limits_{x_i}\frac{1}{2}(\hat{x}_i-x_i)^2+\lambda\vert x_i\vert$$
%
% Global minimum solution for $l_1$-norm minimization can be obtained using
% soft thresholding operator:
%
% $$x_i^*=S(\hat{x},\lambda)=sign(\hat{x}(\vert \hat{x}\vert)-\lambda)_+$$
%
% When the design matrix $A$ is orthogonal, both $l_0$- and $l_1$-norm
% minimization problems decompose into a set of independent univariate
% problems which can be easily solved by first computing the OLS solution
% $\hat{x}=A^Ty$ and then applying thresholding operators to each
% coefficient.

%soft thresholding
softThresholding = @(x, th) sign(x).*max(abs(x)-th,0);

x = -1:0.01:1;
threshold = 0.35;
x_st = softThresholding(x, threshold);

figure, axis tight
plot(x, x_st)
hold on
plot(x, x, '-.')
title('Soft Thresholding Operator')
xlabel('x')
ylabel('x*')

%% Algorithms for $l_0$-norm Minimization
%
% In this section we focus on approximate optimization methods such as
% greedy approaches for solving $l_0$-norm minimization problems:
%
% * $\min\limits_x\Vert x\Vert_0 \quad s.t.\quad \Vert y-Ax\Vert_2\leq\epsilon$
% * $\min\limits_x\Vert y-Ax\Vert_2 \quad s.t.\quad \Vert x\Vert_0\leq k$
%
% In the second form of the optimization problem $k$ represents bound on
% the number of nonzero elements and is uniquely defined by parameter
% $\epsilon$ in the first formulation of the problem. The latter problem is
% also known as the best subset selection problem, since it aims at finding
% a subset of $k$ variables that yield the lowest quadratic loss, i.e. the
% best linear regression fit.
% 
% At high-level greedy algorithmic scheme can be outlined as:
%
% # Start with an empty support set, i.e. the set of nonzero variables ,
% and zero vector as the current solution
% # Select the best variable using some ranking cirterion $C_{rank}$, and
% add the variable to the current support set.
% # Update the current solution, and recompute the current objective
% function, also called the residual.
% # If the current solution $x$ satisfies a given stopping criterion
% $C_{stop}$, exit and return $x$, otherwise go to step 2.
%
% Gready methods:
%
% * Matching Pursuit
% * Orthogonal Matching Pursuit(OMP)
% * Least-squares OMP(LS-OMP)
% * Stagewise OMP(StOMP)
% * Regularized OMP(ROMP)
%% Algorithms for $l_1$-norm Minimization(LASSO)
%
%
% * Least Angle Regression for LASSO(LARS)
% * Coordinate Descent
% * Proximal methods
% * Accelerated methods
%
%
%% Dictionary Learning (Sparse Coding)
% A fixed dictionary may not necessarily be the best match for a particular
% type of signals, since a given basis (columns of $A$, or dictionary elements) 
% may not yield a sufficiently sparse representation of such signals. 
% Thus, a promising alternative approach that became popular in past 
% years is to learn a dictionary that allows for a sparse representation, 
% given a training set of observed signal samples. Given a data matrix $Y$,
% where each column represents an 
% observed signal (sample),we want to find the design matrix, 
% or dictionary, $A$, as well as a sparse representation of each observed 
% signal in that dictionary, corresponding to the sparse columns 
% of the matrix $X$.
% 
% <<dictionary_learning.png>>
% 
% We now formally state the dictionary learning, or sparse coding, problem.
% Let $Y$ be an $n\times N$ matrix, where $n<N$,and the i-th column, 
% or sample, $y_i$, is a vector of observations obtained using 
% linear projections specified by some unknown $n\times K$ matrix D, 
% of the corresponding sparse column-vector $x_i$ of the (unobserved) $K\times N$ matrix $X$.
% For example, if the columns of $Y$ are (vectorized) images,
% such as fMRI scans of a brain, then the columns of $D$ are dictionary elements, or atoms 
% (i.e., some "elementary" images, for example, corresponding 
% to particular brain areas known to be activated by specific 
% tasks and/or stimuli), and the columns of $X$ correspond to sparse 
% codes needed to represent each image using the dictionary (i.e., one can 
% hypothesize that the brain activation observed in a given fMRI scan can 
% be represented as a weighted linear superposition of a relatively small 
% number of active brain areas, out of a potentially large number of such areas).
%
% The ultimate sparse-coding objective is to find both $D$ and $X$ that
% yield the sparsest representation of the data $Y$, subject to some
% acceptable approximation error $\epsilon$:
%
% $$\min\limits_{D,X}\sum\limits_{i=1}^N \Vert x_i\Vert_0 \quad s.t.\quad \Vert Y-DX\Vert_2\leq \epsilon$$
%
% Note that this problem formulation looks very similar 
% to the classical sparse signal recovery problem, only with 
% two modifications:
%
% # dictionary A is now included as an unknown variable that we must optimize over 
% # there are M, rather than just one, observed samples and the corresponding 
% sparse signals, or sparse codes. 
% 
% As usual, there are also two alternative ways of formulating the above
% constrained optimization problem, i.e. by reversing the roles of the
% objective and the constraint:
%
% $$\min\limits_{D,X}\Vert Y-DX\Vert_2^2  \quad s.t. \quad \Vert X(i,:)\Vert_0\leq k, 1\leq i\leq N$$
%
% for some k that corresponds to the above $\epsilon$, or by using the
% Lagrangian relaxation:
%
% $$\min\limits_{D,X}\Vert Y-DX\Vert_2^2 + \lambda\sum\limits_{i=1}^N\Vert X_{i,:}\Vert_0$$
%
% Clearly, the computational complexity of dictionary learning is 
% at least as high as the complexity of the original (NP-hard) 
% $l_0$-norm minimization problem. Thus, the $l_1$-norm relaxation can be 
% applied, as before, to at least convexify the subproblem concerned 
% with optimizing over the $X$ matrix. Also, it is common to constrain 
% the norm of the dictionary elements (e.g., by unit norm), in order to 
% avoid arbitrarily large values of $D$ elements (and, correspondingly, 
% infinitesimal values of $X$ entries) during the optimization process.
%
% $$\min\limits_{D,X}\Vert Y-DX\Vert_2^2 + \lambda\sum\limits_{i=1}^N\Vert X_{i,:}\Vert_1 \quad s.t. 
% \quad \Vert A_{:,j}\Vert_2\leq 1, \quad\forall j =1,...,n$$
%
% There exist several algorithms for dictionary learning:
%
% * Method of Optimal Directions(MOD)
% * K-SVD
% * Online dictionary learning
%% Discriminative Sparse Representation
% If sufficient training samples $Y$ are available from 
% each class, it will be possible to represent the test
% samples as a linear combination of just those training samples from
% the same class. This representation is naturally sparse, involving only
% a small fraction of the overall training database. We argue that in many
% problems of interest, it is actually the sparsest linear representation
% of the test sample in terms of this dictionary and can be recovered
% efficiently via $l_1$-minimization. Seeking the sparsest representation
% therefore automatically discriminates between the various classes
% present in the training set.
%
% A basic problem in object recognition is to use labeled training samples
% from k distinct object classes to correctly determine the class to which
% a new test sample belongs. We arrange the given $n_i$ training samples from
% the $i$ th class as columns of a matrix $D_i=[d_{i,1}, d_{i,2}, ..., d_{i,n_i}]\in 
% \mathbf{R}^{m\times n_i}$
%
% Given sufficient training samples of the $i$th object class,
% $D_i=[d_{i,1}, d_{i,2}, ..., d_{i,n_i}]\in \mathbf{R}^{m\times n_i}$, any
% new test sample $y\in \mathbf{R}^n$ from the same class will
% approximately lie in the linear span of the training samples associated
% with object $i$:
%
% $$y=\alpha_{i,1} d_{i,1} + \alpha_{i,2} d_{i,2} + ... + \alpha_{i,n_i} d_{i,n_i}$$
%
% for some scalars, $\alpha_{i,j}\in\mathbf{R}$, $j=1,2,...n_i$
%
% Since the membership $i$ of the test sample is initially unknown, we
% define a new matrix $D$ for the entire training set as the concatenation
% of the $n$ training samples of all $k$ object clases:
%
% $$D=[D_1, D_2, ..., D_k]=[d_{1,1}, d_{1,2}, ..., d_{k,n_k}]$$
%
% Then, the linear representation of $y$ can be rewritten in terms of all
% training samples as $y=Dx_0\in \mathbf{R}^m$ where 
% $x_0=[0,....,0,\alpha_{i,1},\alpha_{i,2},...,\alpha_{i,n_i},0,...,0]^T\in\mathbf{R}^n$
% is a coefficient vector whose entries are zero except those associated
% with the $i$ th class.
%
% As the entries of the vector $x_0$ encode the identity of the test sample
% $y$, it is tempting to attempt to obtain it by solving the linear system
% of equations $y=Dx$. This can be done by using sparse optimization.
%
% Given a new test sample $y$ from one of the classes in the training set,
% we first compute its sparse representation $\hat{x}_1$ using sparse
% coding. Ideally, the nonzero entries in the estimate $\hat{x}_1$ will all
% be associated with the columns of $D$ from a single object class $i$, and
% we can easily assign the test sample $y$ to that class. However, noise and modeling 
% error may lead to small nonzero entries
% associated with multiple object calsses. We can classify $y$ based on how
% well the coefficients associated with all training samples of each object
% reproduce $y$.
%
% For each calss $i$, let $\delta_i:\mathbf{R}^n\to\mathbf{R}^n$ be the
% characteristic function that selects the coefficients associated with the
% $i$ th class. For $x\in\mathbf{R}^n$, $\delta_i(x)\in\mathbf{R}^n$ is a
% new vector whose only nonzero entries are the entries in $x$ that are
% associated with calss $i$. Using only the coefficients associated with
% the $i$ th class, one can approximate the given test sample $y$ as
% $\hat{y}_i=D\delta_i(\hat{x}_i)$. We then classify $y$ based on these
% approximations by assigning it to the object class that minimizes the
% residual between $y$ and $\hat{y}_i$:
%
% $$\min\limits_i r_i(y)=\Vert y-D\delta_i(\hat{x}_1)\Vert_2$$
%
% 
% <<src.png>>
% 
%% Discriminative K-SVD (D-KSVD)
% Let $Y\in\mathbf{R}^{n\times N}$ denote a set of $N$ $n$-dimensional
% training signals with a corresponding label matrix
% $H\in\mathbf{R}^{m\times N}$, where $m$ is the number of classes. Each
% column $h_i$ of the label matrix $H$ encodes the class label of sample
% $i$ using the position of the nonzero value. For example, if the label of
% sample $y_i$ is 3, then $h_i=[0,0,1,0,...,0]^T$.
%
% The original K-SVD algorithm solves the following optimization problem:
%
% $$\langle D^*, X^*\rangle=\min\limits_{D,X}\Vert Y-DX\Vert^2_F \quad s.t.\quad \Vert x_i\Vert_0\leq T_0, \quad i=1,...,N$$
%
% where $T_0$ is the sparsity constraint, making sure that each sparse
% representation $x_i$ contains not more than $T_0$ nonzero entries. The
% dictionary $D\in\mathbf{n\times K}$, where $K>n$ is the number of atoms
% in the dictionary, and the sparse codes $X\in\mathbf{R}^{K\times N}$,
% obtained by the K-SVD solution minimize the signals reconstruction error
% under the sparsity constraint $T_0$.
%
% The goal of D-KSVD is to use the given label matrix $H$ to learn a linear
% classifier $W\in\mathbf{R}^{m\times K}$ taking in a signals sparse
% representation $x_i$ and returning the most probable class this signal
% belongs to. A straightforward approach to this would be to solve the
% following linear ridge regression problem:
%
% $$W=\min\limits_W\Vert H-WX\Vert_F^2+\lambda\Vert W\Vert_F^2$$
%
% where $\lambda$ is the regularization parameter. This problem has the
% following closed form solution:
%
% $$W=HX^{*T}(X^*X^*T+\lambda I )^-1$$
%
% The drawback of this solution is that learning the classifier $W$ is done
% independently from learning the dictionary $D$ and the sparse codes $X$,
% and thus is suboptimal: the dictionary learning procedure does not take
% into account the fact that its output will be used to train a classifier.
%
% To overcome the sub-optimality of the K-SVD algorithm for classification discussed above,
% one can incorporate the classification error term directly into the 
% K-SVD dictionary learning formulation, causing the K-SVD algorithm to
% simultaneously learn the dictionary and the classifier. 
% The joint dictionary classifier learning problem is defined as follows:
%
% $$\langle D^*, W^*, X^*\rangle=\min\limits_{D,X}\Vert Y-DX\Vert^2_F + \gamma\Vert H-WX\Vert_F^2 \quad s.t.\quad \forall i \Vert x_i\Vert_0\leq T_0$$ 
%
% where $\gamma$ is a regularization parameter balancing the contribution
% of the classification error and discrimination power.
% 
% <<dksvd.png>>
% 




