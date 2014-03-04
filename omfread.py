import numpy
import struct

def read_textfile(filename):
    fid = open(filename,'r')
    oommf_header = [fid.readline()]

    for i in range(1, 34):
        oommf_header.append(fid.readline())

    oommf_data=fid.read()
    mx_my_mz=oommf_data.splitlines()
    fid.close()
    mx_len=len(mx_my_mz)-2

    for i in range(1, 33):
        if oommf_header[i].find('# xnodes:') != -1:
            xnodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# ynodes:') != -1:
            ynodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# znodes:') != -1:
            znodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# xstepsize:') != -1:
            xstepsize = float(oommf_header[i][13:])
        elif oommf_header[i].find('# ystepsize:') != -1:
            ystepsize = float( oommf_header[i][13:])
        elif oommf_header[i].find('# zstepsize:') != -1: 
            zstepsize = float(oommf_header[i][13:])

    mx_my_mz_arr=mx_my_mz[0].split()
    mx_arr=[float(mx_my_mz_arr[0])]
    my_arr=[float(mx_my_mz_arr[1])]
    mz_arr=[float(mx_my_mz_arr[2])]

    for i in range(1, mx_len):
        mx_my_mz_arr=mx_my_mz[i].split()
        mx_arr.append(float(mx_my_mz_arr[0]))
        my_arr.append(float(mx_my_mz_arr[1]))
        mz_arr.append(float(mx_my_mz_arr[2]))

    np_mx = numpy.array(mx_arr)
    np_my = numpy.array(my_arr)
    np_mz = numpy.array(mz_arr)

    mx=numpy.reshape(np_mx,[znodes,ynodes,xnodes])
    my=numpy.reshape(np_my,[znodes,ynodes,xnodes])
    mz=numpy.reshape(np_mz,[znodes,ynodes,xnodes])

    result = [mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize]
    return result

def read_bi4file(filename):
    
    fid = open(filename,'rb')
    oommf_header = [fid.readline()]

    for i in range(1, 34):
        oommf_header.append(fid.readline())

    data=fid.read()
    fid.close()

    for i in range(1, 33):
        if oommf_header[i].find('# xnodes:') != -1:
            xnodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# ynodes:') != -1:
            ynodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# znodes:') != -1:
            znodes = int(oommf_header[i][9:])
        elif oommf_header[i].find('# xstepsize:') != -1:
            xstepsize = float(oommf_header[i][13:])
        elif oommf_header[i].find('# ystepsize:') != -1:
            ystepsize = float( oommf_header[i][13:])
        elif oommf_header[i].find('# zstepsize:') != -1: 
            zstepsize = float(oommf_header[i][13:])

    N=xnodes*ynodes*znodes
    filepos=4
    vector = struct.unpack('!'+'fff'*N,data[ filepos: filepos + 3*4*N])
    vectorfield = numpy.reshape( numpy.array( vector ), (N, 3) )

    mx_arr=vectorfield[:,0]
    my_arr=vectorfield[:,1]
    mz_arr=vectorfield[:,2]

    mx=numpy.reshape(mx_arr,[znodes,ynodes,xnodes])
    my=numpy.reshape(my_arr,[znodes,ynodes,xnodes])
    mz=numpy.reshape(mz_arr,[znodes,ynodes,xnodes])

    result = [mx, my, mz, xnodes, ynodes, znodes, xstepsize, ystepsize, zstepsize]
    return result

