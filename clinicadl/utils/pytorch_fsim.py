### from https://github.com/mikhailiuk/pytorch-fsim

import torch as pt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import math
import imageio
from scipy.io import loadmat

'''
This code is a direct pytorch implementation of the original FSIM code provided by
Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang in Matlab. For the original version
please see: 

https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm

'''


class FSIM_base(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.cuda_computation = False
        self.nscale = 4 # Number of wavelet scales
        self.norient = 4 # Number of filter orientations
        self.k = 2.0 # No of standard deviations of the noise
                     # energy beyond the mean at which we set the
                     # noise threshold point. 
                     # below which phase congruency values get
                     # penalized.

        self.epsilon = .0001 # Used to prevent division by zero
        self.pi = math.pi
        
        minWaveLength = 6  # Wavelength of smallest scale filter
        mult = 2  # Scaling factor between successive filters
        sigmaOnf = 0.55 # Ratio of the standard deviation of the
                        # Gaussian describing the log Gabor filter's
                        # transfer function in the frequency domain
                        # to the filter center frequency.    
        dThetaOnSigma = 1.2 # Ratio of angular interval between filter orientations    
                            # and the standard deviation of the angular Gaussian
                            # function used to construct filters in the
                            # freq. plane.
        
        self.thetaSigma = self.pi/self.norient/dThetaOnSigma # Calculate the standard deviation of the
                                                             # angular Gaussian function used to
                                                             # construct filters in the freq. plane.
        

        self.fo = (1.0/(minWaveLength*pt.pow(mult,(pt.arange(0,self.nscale,dtype=pt.float64))))).unsqueeze(0) # Centre frequency of filter
        self.den = 2*(math.log(sigmaOnf))**2
        self.dx = -pt.tensor([[[[3, 0, -3], [10, 0,-10], [3,0,-3]]]])/16.0
        self.dy = -pt.tensor([[[[3, 10, 3], [0, 0, 0],   [-3 ,-10, -3]]]])/16.0
        self.T1 = 0.85
        self.T2 = 160
        self.T3 = 200;
        self.T4 = 200;
        self.lambdac = 0.03

    def set_arrays_to_cuda(self):
        self.cuda_computation = True
        self.fo = self.fo.cuda()
        self.dx = self.dx.cuda()
        self.dy = self.dy.cuda()
    
    def forward_gradloss(self,imgr,imgd):
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)

        
        #PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        #gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        gradloss = pt.sum(pt.sum(pt.sum(gradientSimMatrix,1),1))
        return gradloss
    
    def calculate_fsim(self,gradientSimMatrix,PCSimMatrix,PCm):
        SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
        FSIM = pt.sum(pt.sum(SimMatrix,1),1) / pt.sum(pt.sum(PCm,1),1)
        return FSIM

    def calculate_fsimc(self, I1,Q1,I2,Q2,gradientSimMatrix,PCSimMatrix,PCm):

        ISimMatrix = (2*I1*I2 + self.T3) / (pt.pow(I1,2) + pt.pow(I2,2) + self.T3)
        QSimMatrix = (2*Q1*Q2 + self.T4) / (pt.pow(Q1,2) + pt.pow(Q2,2) + self.T4)
        SimMatrixC = gradientSimMatrix*PCSimMatrix*(pt.pow(pt.abs(ISimMatrix*QSimMatrix),self.lambdac))*PCm
        FSIMc = pt.sum(pt.sum(SimMatrixC,1),1)/pt.sum(pt.sum(PCm,1),1)

        return FSIMc
    
    def lowpassfilter(self, rows, cols):
        cutoff = .45
        n = 15
        x, y = self.create_meshgrid(cols,rows)
        radius = pt.sqrt(pt.pow(x,2) + pt.pow(y,2)).unsqueeze(0)       
        f = self.ifftshift2d( 1 / (1.0 + pt.pow(pt.div(radius,cutoff),2*n)) ) 
        return f
    
    def calculate_gradient_sim(self,gradientMap1,gradientMap2):

        gradientSimMatrix = (2*gradientMap1*gradientMap2 + self.T2) /(pt.pow(gradientMap1,2) + pt.pow(gradientMap2,2) + self.T2)
        return gradientSimMatrix

    def calculate_gradient_map(self,Y):
        IxY = F.conv2d(Y,self.dx, padding=1)
        IyY = F.conv2d(Y,self.dy, padding=1)
        gradientMap1 = pt.sqrt(pt.pow(IxY,2) + pt.pow(IyY,2))
        return gradientMap1
    
    def calculate_phase_score(self,PC1,PC2):
        PCSimMatrix = (2 * PC1 * PC2 + self.T1) / (pt.pow(PC1,2) + pt.pow(PC2,2) + self.T1)
        PCm = pt.where(PC1>PC2, PC1,PC2)
        return PCSimMatrix,PCm    
    
    def roll_1(self,x, n):  
        return pt.cat((x[:,-n:,:,:,:], x[:,:-n,:,:,:]), dim=1)        
        
    def ifftshift(self,tens,var_axis):
        len11 = int(tens.size()[var_axis]/2)
        len12 = tens.size()[var_axis]-len11
        return pt.cat((tens.narrow(var_axis,len11,len12),tens.narrow(var_axis,0,len11)),axis=var_axis)

    def ifftshift2d(self,tens):
        return self.ifftshift(self.ifftshift(tens,1),2)

    def create_meshgrid(self,cols,rows):
        '''
        Set up X and Y matrices with ranges normalised to +/- 0.5
        The following code adjusts things appropriately for odd and even values
        of rows and columns.
        '''

        if cols%2:
            xrange = pt.arange(start = -(cols-1)/2, end = (cols-1)/2+1, step = 1, requires_grad=False)/(cols-1)
        else:
            xrange = pt.arange(-(cols)/2, (cols)/2, step = 1, requires_grad=False)/(cols)

        if rows%2:
            yrange = pt.arange(-(rows-1)/2, (rows-1)/2+1, step = 1, requires_grad=False)/(rows-1)
        else:
            yrange = pt.arange(-(rows)/2, (rows)/2, step = 1, requires_grad=False)/(rows)

        x, y = pt.meshgrid([xrange, yrange])
        
        if self.cuda_computation:
            x, y = x.cuda(), y.cuda()
            
        return x.T, y.T

    def process_image_channels(self,img):


        batch, rows, cols = img.shape[0],img.shape[2],img.shape[3]

        minDimension = min(rows,cols)    

        Ycoef = pt.tensor([[0.299,0.587,0.114]])
        Icoef = pt.tensor([[0.596,-0.274,-0.322]])
        Qcoef = pt.tensor([[0.211,-0.523,0.312]])
        
        if self.cuda_computation:
            Ycoef, Icoef, Qcoef = Ycoef.cuda(), Icoef.cuda(), Qcoef.cuda()

        Yfilt=pt.cat(batch*[pt.cat(rows*cols*[Ycoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        Ifilt=pt.cat(batch*[pt.cat(rows*cols*[Icoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        Qfilt=pt.cat(batch*[pt.cat(rows*cols*[Qcoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        
        # If images have three chanels
        if img.size()[1]==3:
            Y = pt.sum(Yfilt*img,1).unsqueeze(1)
            I = pt.sum(Ifilt*img,1).unsqueeze(1)
            Q = pt.sum(Qfilt*img,1).unsqueeze(1)
        else:
            Y = pt.mean(img,1).unsqueeze(1)
            I = pt.ones(Y.size(),dtype=pt.float64)
            Q = pt.ones(Y.size(),dtype=pt.float64)

        F = max(1,round(minDimension / 256))

        aveKernel = nn.AvgPool2d(kernel_size = F, stride = F, padding =0)# max(0, math.floor(F/2)))
        if self.cuda_computation:
            aveKernel = aveKernel.cuda()
            
        # Make sure that the dimension of the returned image is the same as the input
        I = aveKernel(I)
        Q = aveKernel(Q)
        Y = aveKernel(Y)
        return I,Q,Y

        
    def phasecong2(self,img):
        '''
        % Filters are constructed in terms of two components.
        % 1) The radial component, which controls the frequency band that the filter
        %    responds to
        % 2) The angular component, which controls the orientation that the filter
        %    responds to.
        % The two components are multiplied together to construct the overall filter.

        % Construct the radial filter components...

        % First construct a low-pass filter that is as large as possible, yet falls
        % away to zero at the boundaries.  All log Gabor filters are multiplied by
        % this to ensure no extra frequencies at the 'corners' of the FFT are
        % incorporated as this seems to upset the normalisation process when
        % calculating phase congrunecy.
        '''

        batch, rows, cols = img.shape[0],img.shape[2],img.shape[3]

        imagefft = pt.rfft(img,signal_ndim=2,onesided=False)

        x, y = self.create_meshgrid(cols,rows)

        radius = pt.cat(batch*[pt.sqrt(pt.pow(x,2) + pt.pow(y,2)).unsqueeze(0)],0)
        theta = pt.cat(batch*[pt.atan2(-y,x).unsqueeze(0)],0)

        radius = self.ifftshift2d(radius) # Matrix values contain *normalised* radius from centre
        theta  = self.ifftshift2d(theta) # Matrix values contain polar angle.
                                         # (note -ve y is used to give +ve
                                         # anti-clockwise angles)

        radius[:,0,0] = 1 
        
        sintheta = pt.sin(theta)
        costheta = pt.cos(theta)

        lp = self.lowpassfilter(rows,cols) # Radius .45, 'sharpness' 15
        lp = pt.cat(batch*[lp.unsqueeze(0)],0)
 
        term1 = pt.cat(rows*cols*[self.fo.unsqueeze(2)],dim=2).view(-1,self.nscale,rows,cols)
        term1 = pt.cat(batch*[term1.unsqueeze(0)],0).view(-1,self.nscale,rows,cols)

        term2 = pt.log(pt.cat(self.nscale*[radius.unsqueeze(1)],1)/term1)
        #  Apply low-pass filter    
        logGabor = pt.exp(-pt.pow(term2,2)/self.den)
        logGabor = logGabor*lp
        logGabor[:,:,0,0] = 0 # Set the value at the 0 frequency point of the filter
                              # back to zero (undo the radius fudge).

        # Then construct the angular filter components...

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        angl = pt.arange(0,self.norient,dtype=pt.float64)/self.norient*self.pi

        if self.cuda_computation:
            angl = angl.cuda()
        ds_t1 = pt.cat(self.norient*[sintheta.unsqueeze(1)],1)*pt.cos(angl).view(-1,self.norient,1,1)
        ds_t2 = pt.cat(self.norient*[costheta.unsqueeze(1)],1)*pt.sin(angl).view(-1,self.norient,1,1)
        dc_t1 = pt.cat(self.norient*[costheta.unsqueeze(1)],1)*pt.cos(angl).view(-1,self.norient,1,1)
        dc_t2 = pt.cat(self.norient*[sintheta.unsqueeze(1)],1)*pt.sin(angl).view(-1,self.norient,1,1)
        ds = ds_t1-ds_t2 # Difference in sine.
        dc = dc_t1+dc_t2 # Difference in cosine.
        dtheta = pt.abs(pt.atan2(ds,dc)) # Absolute angular distance.
        spread = pt.exp(-pt.pow(dtheta,2)/(2*self.thetaSigma**2)) # Calculate the
                                                                  # angular filter component.

        logGabor_rep = pt.repeat_interleave(logGabor,self.norient,1).view(-1,self.nscale,self.norient,rows,cols)

        # Batch size, scale, orientation, pixels, pixels
        spread_rep = pt.cat(self.nscale*[spread]).view(-1,self.nscale,self.norient,rows,cols)
        filter_log_spread = logGabor_rep*spread_rep
        array_of_zeros = pt.zeros(filter_log_spread.unsqueeze(5).size(),dtype=pt.float64)
        if self.cuda_computation:
            array_of_zeros = array_of_zeros.cuda()
        filter_log_spread_zero = pt.cat((filter_log_spread.unsqueeze(5),array_of_zeros), dim=5)
        ifftFilterArray = pt.ifft(filter_log_spread_zero,signal_ndim =2).select(5,0)*math.sqrt(rows*cols)

        imagefft_repeat = pt.cat(self.nscale*self.norient*[imagefft],dim=1).view(-1,self.nscale,self.norient,rows,cols,2)
        filter_log_spread_repeat = pt.cat(2*[filter_log_spread.unsqueeze(5)],dim=5)
        # Convolve image with even and odd filters returning the result in EO
        EO = pt.ifft(filter_log_spread_repeat*imagefft_repeat,signal_ndim=2)

        E = EO.select(5, 0)
        O = EO.select(5, 1)
        An = pt.sqrt(pt.pow(E,2)+pt.pow(O,2))
        sumAn_ThisOrient = pt.sum(An,1)
        sumE_ThisOrient = pt.sum(E,1) # Sum of even filter convolution results
        sumO_ThisOrient = pt.sum(O,1) # Sum of odd filter convolution results.

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = pt.sqrt(pt.pow(sumE_ThisOrient,2) + pt.pow(sumO_ThisOrient,2)) + self.epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy
        
        MeanO = pt.cat(self.nscale*[MeanO.unsqueeze(1)],1)
        MeanE = pt.cat(self.nscale*[MeanE.unsqueeze(1)],1)


        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.
        Energy = pt.sum( E*MeanE+O*MeanO - pt.abs(E*MeanO-O*MeanE),1)
        abs_EO  = pt.sqrt(pt.pow(E,2) + pt.pow(O,2))

        #   % Compensate for noise
        # We estimate the noise power from the energy squared response at the
        # smallest scale.  If the noise is Gaussian the energy squared will have a
        # Chi-squared 2DOF pdf.  We calculate the median energy squared response
        # as this is a robust statistic.  From this we estimate the mean.
        # The estimate of noise power is obtained by dividing the mean squared
        # energy value by the mean squared filter value
        medianE2n = pt.pow(abs_EO.select(1,0),2).view(-1,self.norient,rows*cols).median(2).values

        EM_n = pt.sum(pt.sum(pt.pow(filter_log_spread.select(1,0),2),3),2)
        noisePower = -(medianE2n/math.log(0.5))/EM_n
        
        # Now estimate the total energy^2 due to noise
        # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
        EstSumAn2 = pt.sum(pt.pow(ifftFilterArray,2),1)

        sumEstSumAn2 = pt.sum(pt.sum(EstSumAn2,2),2)
        roll_t1 = ifftFilterArray*self.roll_1(ifftFilterArray,1)
        roll_t2 = ifftFilterArray*self.roll_1(ifftFilterArray,2)
        roll_t3 = ifftFilterArray*self.roll_1(ifftFilterArray,3)
        rolling_mult = roll_t1+roll_t2+roll_t3
        EstSumAiAj = pt.sum(rolling_mult,1)/2
        sumEstSumAiAj = pt.sum(pt.sum(EstSumAiAj,2),2)

        EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2+4*noisePower*sumEstSumAiAj
        tau = pt.sqrt(EstNoiseEnergy2/2)
        EstNoiseEnergy = tau*math.sqrt(self.pi/2)
        EstNoiseEnergySigma = pt.sqrt( (2-self.pi/2)*pt.pow(tau,2))


        # The estimated noise effect calculated above is only valid for the PC_1 measure. 
        # The PC_2 measure does not lend itself readily to the same analysis.  However
        # empirically it seems that the noise effect is overestimated roughly by a factor 
        # of 1.7 for the filter parameters used here.
        T = (EstNoiseEnergy + self.k*EstNoiseEnergySigma)/1.7 # Noise threshold
        
        T_exp = pt.cat(rows*cols*[T.unsqueeze(2)],dim=2).view(-1,self.norient,rows,cols)
        AnAll = pt.sum(sumAn_ThisOrient,1)
        array_of_zeros_energy = pt.zeros(Energy.size(),dtype=pt.float64)
        if self.cuda_computation:
            array_of_zeros_energy =array_of_zeros_energy.cuda()
            
        EnergyAll = pt.sum(pt.where((Energy - T_exp)<0.0, array_of_zeros_energy,Energy - T_exp ),1)
        ResultPC = EnergyAll/AnAll
        
        return ResultPC
    
class FSIM(FSIM_base):
    '''
    Note, the input is expected to be from 0 to 255
    '''

    def __init__(self):
        super().__init__()
        
    def forward(self,imgr,imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()
            
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)
        
        PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        FSIM = self.calculate_fsim(gradientSimMatrix,PCSimMatrix,PCm)

        return FSIM.mean()

class FSIMc(FSIM_base, nn.Module):
    '''
    Note, the input is expected to be from 0 to 255
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self,imgr,imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()
            
            
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)
        
        PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        FSIMc = self.calculate_fsimc(I1.squeeze(),Q1.squeeze(),I2.squeeze(),Q2.squeeze(),gradientSimMatrix,PCSimMatrix,PCm)

        return FSIMc.mean()