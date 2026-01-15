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
  ~/systemic-risk-tax-abm/  58 
   ├╴󰌠  figuras.py  18 
   │ ├╴  Cannot access attribute "values" for class "ndarray[_AnyShape, dtype[Any]]"
   │ │      Attribute "values" is unknown Pyright (reportAttributeAccessIssue) [99, 81]
   │ ├╴  Cannot access attribute "values" for class "ndarray[_AnyShape, dtype[Any]]"
   │ │      Attribute "values" is unknown Pyright (reportAttributeAccessIssue) [120, 54]
   │ ├╴  Operator "*" not supported for types "ArrayLike | Any | Unknown" and "Literal[100]"
   │ │      Operator "*" not supported for types "ExtensionArray" and "Literal[100]" Pyright (reportOperatorIssue) [121, 38]
   │ ├╴  Cannot access attribute "values" for class "ndarray[_AnyShape, dtype[Any]]"
   │ │      Attribute "values" is unknown Pyright (reportAttributeAccessIssue) [121, 59]
   │ ├╴  Argument of type "NDArray[floating[Any]]" cannot be assigned to parameter "bins" of type "int | Sequence[float] | str | None"
   │ │      Type "NDArray[floating[Any]]" is not assignable to type "int | Sequence[float] | str | None"
   │ │        "ndarray[_AnyShape, dtype[floating[Any]]]" is not assignable to "int"
   │ │        "ndarray[_AnyShape, dtype[floating[Any]]]" is not assignable to "Sequence[float]"
   │ │        "ndarray[_AnyShape, dtype[floating[Any]]]" is not assignable to "str"
   │ │        "ndarray[_AnyShape, dtype[floating[Any]]]" is not assignable to "None" Pyright (reportArgumentType) [229, 26]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [59, 26]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [62, 31]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [111, 42]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [139, 20]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [143, 32]
   │ ├╴  Local variable `std_profile` is assigned to but never used Ruff (F841) [149, 9]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [170, 18]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [177, 29]
   │ ├╴  Ambiguous variable name: `l` Ruff (E741) [214, 8]
   │ ├╴  Ambiguous variable name: `l` Ruff (E741) [226, 8]
   │ ├╴  Ambiguous variable name: `l` Ruff (E741) [239, 8]
   │ ├╴  "t" is not accessed Pyright  [109, 21]
   │ └╴  "std_profile" is not accessed Pyright  [149, 9]
   ├╴󰌠  main.py  8 
   │ ├╴  Cannot assign to attribute "MODO_IMPUESTO" for class "type[Param]"
   │ │      Attribute "MODO_IMPUESTO" is unknown Pyright (reportAttributeAccessIssue) [231, 11]
   │ ├╴  Module level import not at top of file Ruff (E402) [39, 1]
   │ ├╴  Module level import not at top of file Ruff (E402) [40, 1]
   │ ├╴  `shutil` imported but unused Ruff (F401) [40, 8]
   │ ├╴  Module level import not at top of file Ruff (E402) [42, 1]
   │ ├╴  Local variable `start_time` is assigned to but never used Ruff (F841) [57, 5]
   │ ├╴  "shutil" is not accessed Pyright  [40, 8]
   │ └╴  "start_time" is not accessed Pyright  [57, 5]
   ├╴󰌠  visualizador_red.py  7 
   │ ├╴  Cannot access attribute "append_data" for class "_BaseReaderWriter"
   │ │      Attribute "append_data" is unknown Pyright (reportAttributeAccessIssue) [101, 20]
   │ ├╴  Cannot access attribute "append_data" for class "_BaseReaderWriter"
   │ │      Attribute "append_data" is unknown Pyright (reportAttributeAccessIssue) [186, 20]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [96, 23]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [103, 24]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [181, 23]
   │ ├╴  Multiple statements on one line (colon) Ruff (E701) [188, 24]
   │ └╴  "fig" is not accessed Pyright  [75, 9]
   └╴  logica/  25 
     ├╴󰌠  paso1.py  2 
     │ ├╴  `parametros.Param` imported but unused Ruff (F401) [2, 33]
     │ └╴  "p" is not accessed Pyright  [2, 33]
     ├╴󰌠  paso2.py  17 
     │ ├╴  `parametros.Param` imported but unused Ruff (F401) [2, 33]
     │ ├╴  Multiple statements on one line (colon) Ruff (E701) [192, 36]
     │ ├╴  Local variable `v_loop` is assigned to but never used Ruff (F841) [210, 17]
     │ ├╴  Multiple statements on one line (colon) Ruff (E701) [224, 37]
     │ ├╴  Local variable `delta` is assigned to but never used Ruff (F841) [231, 17]
     │ ├╴  Multiple statements on one line (colon) Ruff (E701) [236, 38]
     │ ├╴  Local variable `v_new` is assigned to but never used Ruff (F841) [250, 21]
     │ ├╴  Multiple statements on one line (colon) Ruff (E701) [294, 34]
     │ ├╴  Multiple statements on one line (colon) Ruff (E701) [313, 54]
     │ ├╴  "p" is not accessed Pyright  [2, 33]
     │ ├╴  "H_loop" is not accessed Pyright  [203, 13]
     │ ├╴  "v_loop" is not accessed Pyright  [208, 13]
     │ ├╴  "v_loop" is not accessed Pyright  [210, 17]
     │ ├╴  "delta" is not accessed Pyright  [231, 17]
     │ ├╴  "H_new" is not accessed Pyright  [243, 17]
     │ ├╴  "v_new" is not accessed Pyright  [248, 17]
     │ └╴  "v_new" is not accessed Pyright  [250, 21]
     ├╴󰌠  paso3.py  2 
     │ ├╴  Local variable `wage_values` is assigned to but never used Ruff (F841) [44, 5]
     │ └╴  "wage_values" is not accessed Pyright  [44, 5]
     ├╴󰌠  paso4.py  2 
     │ ├╴  `parametros.Param` imported but unused Ruff (F401) [2, 33]
     │ └╴  "p" is not accessed Pyright  [2, 33]
     └╴󰌠  paso5_6_7.py  2 
       ├╴  `parametros.Param` imported but unused Ruff (F401) [2, 33]
       └╴  "p" is not accessed Pyright  [2, 33]
