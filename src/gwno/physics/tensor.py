"""MF6 constitutive tensor and independent periodic finite-volume verification.

The face operator is not an MF6 XT3D reproduction or field-grid operator.
Concentration axes are (..., z, y, x); tensor and spacing order is (x, y, z).
Uniform porosity cancels from this per-pore-volume diffusion operator.
"""
import torch


def modflow_dispersion_tensor(specific_discharge,porosity,alh,ath1,
                             alv=None,ath2=None,atv=None,diffusion=0.):
    """MF6 6.8.0 constitutive tensor, before geometric XT3D reconstruction.

    Input q is Darcy specific discharge (..., x/y/z), NOT pore velocity.
    Output is the bulk coefficient alpha*|q| + porosity*diffusion, in L^2/T.
    Saturated-cell masking belongs to the caller. No face interpolation here.
    """
    q=specific_discharge
    cast=lambda value: torch.as_tensor(value,dtype=q.dtype,device=q.device)
    alh,ath1=cast(alh),cast(ath1)
    alv=alh if alv is None else cast(alv)
    ath2=ath1 if ath2 is None else cast(ath2)
    atv=ath2 if atv is None else cast(atv)
    speed=torch.linalg.vector_norm(q,dim=-1)
    horizontal=torch.linalg.vector_norm(q[...,:2],dim=-1)
    safe_speed=torch.where(speed>0,speed,torch.ones_like(speed))
    safe_horizontal=torch.where(horizontal>0,horizontal,torch.ones_like(horizontal))
    longitudinal=q/safe_speed[...,None]
    transverse=torch.stack((-q[...,1]/safe_horizontal,
        torch.where(horizontal>0,q[...,0]/safe_horizontal,torch.ones_like(horizontal)),torch.zeros_like(horizontal)),dim=-1)
    third=torch.linalg.cross(longitudinal,transverse,dim=-1)
    vertical_fraction=(q[...,2]/safe_speed)**2
    al=alh*(1-vertical_fraction)+alv*vertical_fraction
    at1=ath1*(1-vertical_fraction)+atv*vertical_fraction
    at2=ath2*(1-vertical_fraction)+atv*vertical_fraction
    result=sum((a*speed)[...,None,None]*e[..., :,None]*e[...,None,:]
               for a,e in [(al,longitudinal),(at1,transverse),(at2,third)])
    return result+(cast(porosity)*cast(diffusion))[...,None,None]*torch.eye(3,dtype=q.dtype,device=q.device)


def periodic_tensor_outward(concentration, dispersion, spacing):
    """Return divergence(-D grad C), in concentration/time, for constant D.

    Face-normal derivatives use adjacent cells; tangential derivatives average
    the two centered cell derivatives. A face flux is shared by both cells.
    """
    axes=(-1,-2,-3)
    gradients=[(torch.roll(concentration,-1,a)-torch.roll(concentration,1,a))/(2*h)
               for a,h in zip(axes,spacing)]
    outward=torch.zeros_like(concentration)
    for i,(axis,h) in enumerate(zip(axes,spacing)):
        flux=torch.zeros_like(concentration)
        for j in range(3):
            derivative=((torch.roll(concentration,-1,axis)-concentration)/h if i==j
                        else (gradients[j]+torch.roll(gradients[j],-1,axis))/2)
            flux=flux-dispersion[i,j]*derivative
        outward=outward+(flux-torch.roll(flux,1,axis))/h
    return outward
