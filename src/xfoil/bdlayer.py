import pandas as pd
import numpy as np

class BoundaryLayer():
    """Read and process the boundary layer file created by Xfoil. 

    Note
    ----
    It assumes the file already exists when it is instantiated.
    """

    def __init__(self, filename):

        self.filename = filename

        # Read the file
        self._read_file()

        # Postprocess
        self._find_airfoil()
        self._trim_sections()

    def _read_file(self):
        """Read the boundary layer file and 
        extract the data in the right format.
        """
        # Read the boundary data file
        df_bl = pd.read_csv(self.filename, delim_whitespace=True)
        
        # Get all the variables
        self.variables = df_bl.columns

        # Remove the columns that do not contain data
        df_bl.dropna(axis=1, how='all', inplace=True)

        # Rename the columns appropriately
        df_bl.columns = self.variables[1:len(df_bl.columns)+1]

        # Extract all the variables
        self.s = df_bl['s'].values
        self.x = df_bl['x'].values
        self.y = df_bl['y'].values
        
        self.ue_vinfty = df_bl['Ue/Vinf'].values
        self.d_star    = df_bl['Dstar'].values
        self.theta     = df_bl['Theta'].values

        self.cf = df_bl['Cf'].values

        self.H = df_bl['H'].values
        self.H_star = df_bl['H*'].values
        
        self.P = df_bl['P'].values
        self.m = df_bl['m'].values

        # Memory clean-up
        del df_bl

    def _find_airfoil(self):
        """Find the array index at the end of the profile.  
        
        Notes
        -----
        Creates attribute last_index.
        """
        self.last_index = int()
        pass

    def _trim_sections(self):
        """Cut the BL properties between airfoil and wake. 
        """
        pass