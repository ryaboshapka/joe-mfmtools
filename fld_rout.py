import numpy
def losqual(src_m, xnodes, ynodes, factor):

    dim_x = xnodes//factor
    dim_y = ynodes//factor
    targ_m = numpy.zeros([dim_y, dim_x],dtype=numpy.float)

    for i in range(0, ynodes, factor):
        for j in range(0, xnodes, factor):
            targ_m[i//factor,j//factor]=src_m[i,j]
    return [targ_m, dim_x, dim_y] 
