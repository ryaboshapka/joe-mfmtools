import numpy
import fld_rout 

def d2fz_dz2 (mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize, Z):
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
                105*(mx[k,:,:]*(X0-X)+my[k,:,:]*(Y0-Y)+mz[k,:,:]*(Z0+Z))*(Z0+Z)**3
                /((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(9/2)
                -45*mz[k,:,:]*(Z0+Z)**2/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                -45*(mx[k,:,:]*(X0-X)+my[k,:,:]*(Y0-Y)+mz[k,:,:]*(Z0+Z))*(Z0+Z)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                +9*mz[k,:,:]/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)))
    return F

def lost_d2fz_dz2 (mx, my, mz, nx, ny, nz, xstepsize, ystepsize, zstepsize, Z, factor):
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
                105*(mx2*(X0-X)+my2*(Y0-Y)+mz2*(Z0+Z))*(Z0+Z)**3
                /((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(9/2)
                -45*mz2*(Z0+Z)**2/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                -45*(mx2*(X0-X)+my2*(Y0-Y)+mz2*(Z0+Z))*(Z0+Z)/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(7/2)
                +9*mz2/((X-X0)**2+(Y-Y0)**2+(Z+Z0)**2)**(5/2)))
    return [F*factor*factor, xnodes, ynodes]

def f_d2fz_dz2 (mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize, Z):
    X, Y = numpy.meshgrid(range(-xnodes//2, xnodes//2), range(-ynodes//2, ynodes//2))
    X=X*xstepsize
    Y=Y*ystepsize
    F=numpy.zeros([ynodes,xnodes],dtype=numpy.float)
    for k in range(0, znodes):
        Z0 = k*zstepsize
        fmx=numpy.fft.fft2(mx[k,:,:])
        fmy=numpy.fft.fft2(my[k,:,:])
        fmz=numpy.fft.fft2(mz[k,:,:])

#mx* 105*x*z**3/r2**(9/2)-45*x*z/r2**(7/2)
#my* 105*y*z**3/r2**(9/2)-45*y*z/r2**(7/2)
#mz* 105*z**4/r2**(9/2)-90*z**2/r2**(7/2)+9/r2**(5/2)

        fx=numpy.fft.fft2(105*X*(Z+Z0)**3/(X**2+Y**2+(Z+Z0)**2)**(9/2)-45*X*(Z+Z0)/(X**2+Y**2+(Z+Z0)**2)**(7/2))
        fy=numpy.fft.fft2(105*Y*(Z+Z0)**3/(X**2+Y**2+(Z+Z0)**2)**(9/2)-45*Y*(Z+Z0)/(X**2+Y**2+(Z+Z0)**2)**(7/2))
        fz=numpy.fft.fft2(105*(Z+Z0)**4/(X**2+Y**2+(Z+Z0)**2)**(9/2)-90*(Z+Z0)**2/(X**2+Y**2+(Z+Z0)**2)**(7/2)+9/(X**2+Y**2+(Z+Z0)**2)**(5/2))
   
        F=F+(numpy.fft.ifft2(fmx*fx+fmy*fy+fmz*fz)).real
    
    F1=numpy.zeros([2*ynodes,2*xnodes],dtype=numpy.float)
    F1[0:ynodes,0:xnodes]=F
    F1[0:ynodes,xnodes:2*xnodes]=F
    F1[ynodes:2*ynodes,0:xnodes]=F
    F1[ynodes:2*ynodes,xnodes:2*xnodes]=F
    F=(F1[ynodes//2:3*ynodes//2,xnodes//2:3*xnodes//2]).swapaxes(0,1)
    
    return F
