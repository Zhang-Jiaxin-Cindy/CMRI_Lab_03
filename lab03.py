"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim(provider of the frame) Zhang(implement all methods)
- Email           : <jinho.kim@fau.de>
"""

import numpy as np
import utils
from skimage.filters import window


class Lab03_op:
    def __init__(self, PF):
        """
        PF: Partial Fourier factor. (float)
            This factor is used to calculate the original k-space size.
        """
        self.PF = PF

    def load_kdata_pf(self):
        mat = utils.load_data("kdata_phase_error_severe.mat")
        return mat["kdata"]

    def load_kdata_full(self):
        mat = utils.load_data("kdata1.mat")
        return mat["kdata1"]

    def get_half_zf_kdata(self, kdata: np.ndarray):
        """
        This function returns a half zero-filled kdata of its original shape.

        @param:
            kdata:          k-space data. (shape of [N, M])
        @return:
            zf_kdata:       Half zero-filled kdata. (shape of [N, N])
        """
        # Your code here ...
        zf_kdata = np.zeros((kdata.shape[0], kdata.shape[0]), dtype = complex)
        zf_kdata[:, 0:int(kdata.shape[0]/2)] = kdata[:, 0:int(kdata.shape[0]/2)]
        print(kdata.shape)

        return zf_kdata

    def hermitian_symmetry(self, zf_kdata: np.ndarray):
        """
        This function returns the Hermitian symmetry of the zero-filled k-space data without phase correction.

        @param:
            zf_kdata:       Zero-filled k-space data. (shape of [N, N])
        @return:
            hm_kdata:       Hermitian symmetric kdata. (shape of [N, N])
        """
        # Your code here ...
        
        zf_kdata_half = zf_kdata[:, 0: int(zf_kdata.shape[0]/2)]
        con_kdata_half = np.conj(np.flipud(np.fliplr(zf_kdata_half)))
        
        hm_kdata = np.concatenate((zf_kdata_half, con_kdata_half), axis = 1)

        return hm_kdata

    def estim_phs(self, kdata):
        """
        Phase estimation

        @Param:
            kdata:             asymmetric k-space data (shape of [N, M])
        @Return:
            estimated_phase:    estimated phase of the input kdata
        """
        # Your code here ...
        
        start = kdata.shape[0] - kdata.shape[1]
        end = kdata.shape[1]
        
        sym_area = np.zeros((kdata.shape[0], kdata.shape[0]), dtype = complex)
        sym_area[:, start:end] = kdata[:, start:end]
        
        hamming = np.zeros((kdata.shape[0], kdata.shape[0]), dtype = complex)
        hamming[:, start:end] = window('hamming', (kdata.shape[0], (end - start)))
        estimated_phase = np.angle(utils.ifft2c(hamming * sym_area))

        return estimated_phase

    def get_window(self, kdata, type="ramp"):
        """
        This function returns the window for the Hermitian symmetric extension

        @Param:
            kdata:          asymmetric k-space data
            type:           filter type ('ramp' or 'hamm')
        @Return:
            window_filter:  Window filter for the Hermitian symmetric extension
        """
        # Your code here ...
        # The "width" of the ramp filter is the width of the certer/symmetric kdata(rows)
        # The ramp filter is actually a phase gradient, where the first line will be the zero, and the last will be 1...
        # only in direction of row(from top to bottom)
        window_filter = np.zeros((kdata.shape[0], kdata.shape[0]))
        window_filter[:, int(kdata.shape[0]/2):] = 1
        
        start = kdata.shape[0] - kdata.shape[1]
        end = kdata.shape[1]
        width = end - start
        
        if(type == 'ramp'):
            for i in range(width):
                window_filter[:, i + start] = i / width
        elif(type == 'hamm'):
            hamming = window('hamming',  2*width)
            for i in range(width):
                window_filter[:, i + start] = hamming[i]
        else:
            print("type has no attribute to ", type)
            return 0
        
        window_filter = window_filter * 256
        
        return window_filter

    def pf_margosian(self, kdata, wtype, **kwargs):
        """
        Margosian reconstruction for partial Fourier (PF) MRI

        Param:
            kdata:      asymmetric k-space data
            wtype:      The type of window ('ramp' or 'hamm')
        Return:
            I: reconstructed magnitude image
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'estim_phs' and 'get_window' in this method if you need to used them instead of calling them by self.estim_phs and self.get_window.
        estim_phs = kwargs.get("estim_phs", self.estim_phs)
        get_window = kwargs.get("get_window", self.get_window)

        # Your code here ...
        
        phs = estim_phs(kdata)
        window = get_window(kdata, type = wtype)
        
        zero_padding = np.zeros((kdata.shape[0], kdata.shape[0]), dtype = complex)
        zero_padding[:, 0:kdata.shape[1]] = kdata
        
        I0 = zero_padding*window
        I = np.abs(utils.ifft2c(I0))*np.exp(1j*(np.angle(utils.ifft2c(I0)) - phs))

        return I.real

    def pf_pocs(self, kdata, Nite, **kwargs):
        """
        POCS reconstruction for partial Fourier (PF) MRI

        Param:
            kdata:      asymmetric k-space data
            Nite:       number of iterations

        Return:
            I: reconstructed magnitude image
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'estim_phs' in this method if you need to used it instead of calling it by self.estim_phs.
        estim_phs = kwargs.get("estim_phs", self.estim_phs)

        # Your code here ...
        # initialization
        S0 = np.zeros((kdata.shape[0], kdata.shape[0]), dtype = complex)
        phs = estim_phs(kdata)
        S0[:, 0:kdata.shape[1]] = kdata
        
        for times in range(Nite):
            I0 = utils.ifft2c(S0)
            I1 = np.abs(I0)*np.exp(1j*phs)
            S1 = utils.fft2c(I1)
            S0[:, kdata.shape[1]+1:] = S1[:, kdata.shape[1]+1:]
        
        I = np.abs(I1)

        return I


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab03_solution import *

    # %% Define the lab03 object and load kdata.
    ## The partial Fourier factor is 9/16.
    PF = 9 / 16
    op = Lab03_op(PF)
    kdata = op.load_kdata_pf()
