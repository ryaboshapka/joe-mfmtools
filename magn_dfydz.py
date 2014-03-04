import numpy
import fld_rout 

def dfy_dz (mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize, Z):
    X, Y = numpy.meshgrid(range(-xnodes//2, xnodes//2), range(-ynodes//2, ynodes//2))
    X=X*xstepsize
    Y=Y*ystepsize
    F=numpy.zeros([xnodes,ynodes],dtype=numpy.float)
    for k in range(0, znodes):
        Z0 = k*zstepsize
        for i in range(0, xnodes):
            X0 = (i-xnodes//2)*xstepsize
            for j in range(0, ynodes):
                Y0 = (j-ynodes//2)*ystepsize
                F[i,j]=F[i,j]+sum(sum(
                -15*(mx[k,:,:]*(X0-X)+my[k,:,:]*(Y0-Y)+mz[k,:,:]*(Z0+Z))*(Y0-Y)*(Z0+Z)
                /((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                +3*my[k,:,:]*(Z0+Z)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)
                +3*mz[k,:,:]*(Y0-Y)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)))
    return F

def lost_dfy_dz (mx, my, mz, nx, ny, nz, xstepsize, ystepsize, zstepsize, Z, factor):
    xnodes=nx//factor
    ynodes=ny//factor
    znodes=nz
    X, Y = numpy.meshgrid(range(-xnodes//2, xnodes//2), range(-ynodes//2, ynodes//2))
    X=X*xstepsize*factor
    Y=Y*ystepsize*factor
    F=numpy.zeros([xnodes,ynodes],dtype=numpy.float)
    for k in range(0, znodes):
        Z0 = k*zstepsize
        [mx2, xnodes, ynodes] = fld_rout.losqual(mx[k,:,:], nx, ny, factor)
        [my2, xnodes, ynodes] = fld_rout.losqual(my[k,:,:], nx, ny, factor)
        [mz2, xnodes, ynodes] = fld_rout.losqual(mz[k,:,:], nx, ny, factor)
        for i in range(0, xnodes):
            X0 = (i-xnodes//2)*xstepsize*factor
            for j in range(0, ynodes):
                Y0 = (j-ynodes//2)*ystepsize*factor
                F[i,j]=F[i,j]+sum(sum(
                -15*(mx2*(X0-X)+my2*(Y0-Y)+mz2*(Z0+Z))*(Y0-Y)*(Z0+Z)
                /((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                +3*my2*(Z0+Z)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)
                +3*mz2*(Y0-Y)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)))
    return [F*factor*factor, xnodes, ynodes]

def f_dfy_dz (mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize, Z):
    X, Y = numpy.meshgrid(range(-xnodes//2, xnodes//2), range(-ynodes//2, ynodes//2))
    X=X*xstepsize
    Y=Y*ystepsize
    F=numpy.zeros([ynodes,xnodes],dtype=numpy.float)
    for k in range(0, znodes):
        Z0 = k*zstepsize
        fmx=numpy.fft.fft2(mx[k,:,:])
        fmy=numpy.fft.fft2(my[k,:,:])
        fmz=numpy.fft.fft2(mz[k,:,:])

#mx* -15*x*z*y/r2**(7/2)
#my* -15*z*y**2/r2**(7/2)+3*z/r2**(5/2)
#mz* -15*z**2*y/r2**(7/2)+3*y/r2**(5/2)

        fx=numpy.fft.fft2(-15*X*Y*(Z+Z0)/(X**2+Y**2+(Z+Z0)**2)**(7/2))
        fy=numpy.fft.fft2(-15*Y*Y*(Z+Z0)/(X**2+Y**2+(Z+Z0)**2)**(7/2)+3*(Z+Z0)/(X**2+Y**2+(Z+Z0)**2)**(5/2))
        fz=numpy.fft.fft2(-15*Y*(Z+Z0)**2/(X**2+Y**2+(Z+Z0)**2)**(7/2)+3*Y/(X**2+Y**2+(Z+Z0)**2)**(5/2))
   
        F=F+(numpy.fft.ifft2(fmx*fx+fmy*fy+fmz*fz)).real
    
    F1=numpy.zeros([2*ynodes,2*xnodes],dtype=numpy.float)
    F1[0:ynodes,0:xnodes]=F
    F1[0:ynodes,xnodes:2*xnodes]=F
    F1[ynodes:2*ynodes,0:xnodes]=F
    F1[ynodes:2*ynodes,xnodes:2*xnodes]=F
    F=(F1[ynodes//2:3*ynodes//2,xnodes//2:3*xnodes//2]).swapaxes(0,1)
    
    return F
