# systemic-risk-tax-abm

En principio voy a organizarlo de la siguientte manera:

* output/ -> carpeta donde se guardan los resultados de las simulaciones.
* main.py -> script principal que ejecuta las simulaciones.
* CRISIS/ -> Carpeta con los 7 pasos iterativos, cada uno en un script diferente.
* parametros.py -> archivo con los parámetros del modelo.

Esta es la estructura inicial.

Quiro que se generen las gráficas 3b: R_i vs i para los tres casos, de barras no superpuestas. Y la gráfica 3d: relative $\bigtriangledown EL^{syst} [%]$ vs relative loan size[%], scatter points. Descripcción:FIG. 3. Expected systemic loss as measured by DebtRank, ELsyst
i ∝ Ri. (a) DebtRank, Ri of the 20 largest banks of the
Austrian banking sector at the end of the first quarter of 2006. Banks are ordered by DebtRank, the most important being to
the very left. Inset: Expected systemic loss from all banks for the Austrian interbank data and the three model modes. Here
the SR measure is the size of a potential loss for the entire economy times the probability of that loss occurring as defined
in eq. (3). (b) Model results for Ri: without a tax (red), with the FTT (blue), and with the SRT (green). Clearly, the SRT
drastically reduces the SR contributions of individual banks. The situation without tax resembles the empirical distribution.
(c) Marginal contributions on expected systemic loss ∆(+mn)ELsyst of individual interbank liabilities Lmn vs. the relative size
of interbank loans in double logarithmic scale. Every data point represents an interbank liability L
data
mn , see appendix C. The
loan size captures the credit risk for lenders, whereas ∆(+mn)ELsyst is the SR of the liability. (d) Marginal contributions for
the simulations in the three modes. The SRT reduces SR but leaves contract sizes unchanged.

Y también las figuras 4: todas son de barras no superpuestas para los tres casos.
* Figura 4a: frecuencia vs total losses to banks
* Figura 4b: frecuencia vs cascade sizes
* Figura 4c: frecuencia vs transaction volume IB market

Descripcción: FIG. 4. Comparison of no financial transaction tax (red) on
interbank loans, with systemic risk tax (green), and Tobin tax
(blue). (a) Distribution of total losses to banks L, (b) distribution of cascade sizes C of defaulting banks, and (c) distribution of total transaction volume in the interbank market
V. 10, 000 independent, identical simulations, each with 500
time steps, 20 banks.
