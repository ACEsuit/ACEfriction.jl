using ACEfriction
using ACEfriction.DataUtils: load_h5fdata, save_h5fdata
using Test
using ACEbase.Testing

@info "Test HDF5 import and export of friction data."  

fname = "/test/test-data-100"
filename = string(pkgdir(ACEfriction),fname,".h5")

filename2 = string(tempname(),".h5")
data1 = ACEfriction.DataUtils.load_h5fdata(filename); 
save_h5fdata(data1,filename2);
data2= load_h5fdata(filename2);
rm(filename2)

@test all([all([getfield(d1.atoms,f) == getfield(d2.atoms,f) && 
typeof(getfield(d1.atoms,f)) == typeof(getfield(d2.atoms,f)) for f in fieldnames(typeof(d1.atoms))])
for (d1,d2) in zip(data1,data2)])