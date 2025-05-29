import nrrd

data, header = nrrd.read('../DATA/Dongyang/D1/D1.seg.nrrd')
print(data.shape)
