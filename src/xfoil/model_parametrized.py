"""
+------------------+------------------------------------------------+
| PARSEC parameter | Definition                                     |
+------------------+------------------------------------------------+
| p1               | Leading edge radius                            |
+------------------+------------------------------------------------+
| p2               | Upper crest position in horizontal coordinates |
+------------------+------------------------------------------------+
| p3               | Upper crest position in vertical coordinates   |
+------------------+------------------------------------------------+
| p4               | Upper crest curvature                          |
+------------------+------------------------------------------------+
| p5               | Lower crest position in horizontal coordinates |
+------------------+------------------------------------------------+
| p6               | Lower crest position in vertical coordinates   |
+------------------+------------------------------------------------+
| p7               | Lower crest curvature                          |
+------------------+------------------------------------------------+
| p8               | Trailing edge offset in vertical sense         |
+------------------+------------------------------------------------+
| p9               | Trailing edge thickness                        |
+------------------+------------------------------------------------+
| p10              | Trailing edge direction                        |
+------------------+------------------------------------------------+
| p11              | Trailing edge wedge angle                      |
+------------------+------------------------------------------------+
(Generated with http://www.tablesgenerator.com/)

Source
------    
Sobieczky H. (1999) Parametric Airfoils and Wings. 
In: Fujii K., Dulikravich G.S. (eds) Recent Development of Aerodynamic Design Methodologies. 
Notes on Numerical Fluid Mechanics (NNFM), vol 65. Vieweg+Teubner Verlag
"""


from copy import deepcopy

import numpy as np
from numpy.linalg import solve

from xfoil import Airfoil

class Parsec():
    """
    Parsec airfoil parametrization.

    Parameters
    ----------
    design_vector: dict
    """

    def __init__(self, design_vector: dict = None, N = 160):

        # Save design vector
        if design_vector is None:
            self.dv = self._load_basic_design_vector()
        else:
            self.dv = design_vector

        self.N = N

        # Create specific PARSEC coefficient vectors
        self._q1 = [(2.0 * n - 1)/2.0 for n in range(1,7)]
        self._q2 = [-1.0 / 4,
                     3.0 / 4,
                    15.0 / 4,
                    15.0 / 4,
                    63.0 / 4,
                    99.0 / 4]

    def _load_basic_design_vector(self):
        """
        Load a basic design vector.

        Parameters
        ----------
        None

        Returns
        -------
        dict
        """
        coeff = [
                0.05,    # p1  | Leading edge radius

                0.3,      # p2  | Upper crest position in horizontal coordinates
                0.1,      # p3  | Upper crest position in vertical coordinates
                1.0,      # p4  | Upper crest curvature
                   
                0.3,      # p5  | Lower crest position in horizontal coordinates
                -0.1,     # p6  | Lower crest position in vertical coordinates
                0.0,      # p7  | Lower crest curvature

                0.0,      # p8  | Trailing edge offset in vertical sense
                0.0,      # p9  | Trailing edge thickness
                10.0,     # p10 | Trailing edge direction
                20.0      # p11 | Trailing edge wedge angle
                ]

        return {'p{}'.format(q):value for q, value in zip(range(1,12), coeff)}

    def create_airfoil(self, p:dict = None):
        """
        Create airfoil coordinates compatible with xfoil.

        Parameters
        ----------
        p: dict
            Default: None
            New design vector

        Returns
        -------
        airfoil: xfoil.Airfoil type object
        """
        if p is not None:
            self.dv = deepcopy(p)
        else:
            pass

        self._create_matrices()

        self._rhs_upper = self._create_rhs(sign = '+')
        self._rhs_lower = self._create_rhs(sign = '-')

        self._compute_coefficients()

        self._compute_coordinates()

    def _compute_coefficients(self):

        self._coeff_upper = solve(self._matrix_upper, self._rhs_upper)
        self._coeff_lower = solve(self._matrix_lower, self._rhs_lower)

    def _compute_coordinates(self):

        x = self._cosine_distribution()

        x_q = np.array([x**q for q in self._q1])
        
        self.y_upper = np.dot(np.transpose(self._coeff_upper), x_q)[0]
        self.y_lower = np.dot(np.transpose(self._coeff_lower), x_q)[0]

        self.x = x

    def _cosine_distribution(self):

        from math import cos, pi

        N = self.N

        x = np.linspace(start = 0, stop = 1, num = N)
        x = list(map(cos, x * pi))
        x = np.sort(np.array(x))
        x += 1.0
        x /= 2.0

        return x

    def _create_rhs(self, sign = '+'):

        from math import tan, sqrt, pi

        deg_to_rad = pi / 180.0
        vec = np.ndarray(shape=(6,1))

        if sign == '+':
            vec[0] = self.dv['p8'] + self.dv['p9'] / 2.0
            vec[1] = self.dv['p3']
            vec[2] = tan(deg_to_rad * (self.dv['p10'] - self.dv['p11'] / 2.0))
            vec[3] = 0.0
            vec[4] = self.dv['p4']
            vec[5] = sqrt(2.0 * self.dv['p1'])
        elif sign == '-':
            vec[0] = self.dv['p8'] - self.dv['p9'] / 2.0
            vec[1] = self.dv['p6']
            vec[2] = tan(deg_to_rad * (self.dv['p10'] + self.dv['p11'] / 2.0))
            vec[3] = 0.0
            vec[4] = self.dv['p7']
            vec[5] = -sqrt(2.0 * self.dv['p1'])
        else:
            raise ValueError('Fix the sign!')

        return vec

    def _create_matrices(self):

        self._matrix_upper = self._fill_in_matrix(p = self.dv['p2'])
        self._matrix_lower = self._fill_in_matrix(p = self.dv['p5'])

    def _fill_in_matrix(self, p:float):
        """
        Fill-in matrix according to parameter p2/p5.

        Parameters
        ----------
        p: float
            Crest position in horizontal coordinates
                - Upper: p2
                - Lower: p5

        Returns
        -------
        matrix: np.ndarray(shape=(6,6))

        """

        matrix = np.ndarray(shape=(6,6))

        matrix[0]   = np.ones(6,                          dtype=float)
        matrix[1]   = np.array([p**q for q in self._q1], dtype=float)
        matrix[2]   = np.array([q for q in self._q1],     dtype=float)
        matrix[3]   = np.array([q * p**(q-1.0) for q in self._q1], dtype=float)
        matrix[4]   = np.array([q2 * p**(q1-2.0) for q1, q2 in zip(self._q1, self._q2)], dtype=float)
        matrix[5]   = np.zeros(6, dtype = float)
        matrix[5,0] = 1.0

        return matrix
