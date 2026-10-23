"""Padded spectral Gaussian K fields with varying scenario statistics."""
import numpy as np
from scipy.stats import qmc
CONFIG=dict(base_seed=20260908,geometric_mean_m_day=[.1,1.],log_std=[.5,1.5],horizontal_lengths_m=[300.,1500.],axis_ratio=[1.,3.],angle_deg=[0.,180.],layer_correlation='25 independent; 75 latent rho in [0.3,0.9]',distribution='lognormal, active-cell log moments standardized per layer; no K clipping',covariance='rotated Gaussian spectrum, padding 6 major lengths',parameter_sampling='100-point Latin hypercube; geometric mean sampled logarithmically')

def generate_k(model,scenario,*,seed=None,length_multiplier=1.):
    index=int(scenario.split('_')[-1]);seed=CONFIG['base_seed']+index if seed is None else seed
    if length_multiplier<=0:raise ValueError('K length multiplier must be positive')
    params=qmc.LatinHypercube(d=8,seed=CONFIG['base_seed']).random(100)[index]
    means=10**(-1+params[:2]);sigmas=.5+params[2:4]
    major=300+1200*params[4];minor=max(300.,major/(1+2*params[5]));angle=np.pi*params[6]
    major*=length_multiplier;minor*=length_multiplier
    rho=0. if index%4==0 else .3+.6*params[7]
    rng=np.random.default_rng(seed)
    dx=model.dis.delr.array;dy=model.dis.delc.array
    if not (np.allclose(dx,dx[0]) and np.allclose(dy,dy[0])):
        raise ValueError('Spectral K generation requires supported uniform grid')
    px=int(np.ceil(6*major/dx[0]));py=int(np.ceil(6*major/dy[0]))
    ny,nx=len(dy)+2*py,len(dx)+2*px
    ky=2*np.pi*np.fft.fftfreq(ny,d=dy[0])[:,None]
    kx=2*np.pi*np.fft.rfftfreq(nx,d=dx[0])[None,:]
    u=kx*np.cos(angle)+ky*np.sin(angle);v=-kx*np.sin(angle)+ky*np.cos(angle)
    amplitude=np.exp(-.25*((major*u)**2+(minor*v)**2))
    scale=np.sqrt(np.fft.irfft2(amplitude**2,s=(ny,nx))[0,0])
    def sample():
        field=np.fft.irfft2(np.fft.rfft2(rng.standard_normal((ny,nx)))*amplitude,s=(ny,nx))/scale
        return field[py:py+len(dy),px:px+len(dx)]
    first=sample();second=rho*first+np.sqrt(1-rho*rho)*sample()
    mask=model.dis.idomain.array>0;result=np.empty(mask.shape);actual=[]
    for layer,z in enumerate([first,second]):
        z=(z-z[mask[layer]].mean())/z[mask[layer]].std()
        result[layer]=np.exp(np.log(means[layer])+sigmas[layer]*z)
        values=result[layer][mask[layer]]
        actual.append(dict(geometric_mean=float(np.exp(np.log(values).mean())),log_std=float(np.log(values).std()),min=float(values.min()),max=float(values.max())))
    return result,dict(**CONFIG,seed=seed,layer_geometric_mean=means.tolist(),layer_log_std=sigmas.tolist(),major_length_m=float(major),minor_length_m=float(minor),orientation_deg=float(np.degrees(angle)),latent_layer_rho=float(rho),actual_layer_statistics=actual)
