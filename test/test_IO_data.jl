using ACEfriction
using ACEfriction.DataUtils: load_h5fdata, save_h5fdata
using AtomsBase: position, atomic_number, cell_vectors, periodicity
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

# Compare the atomic systems via the AtomsBase interface (positions, species, cell,
# pbc) rather than raw struct fields: AtomsBase `Atom` has no value-based `==`, so a
# field-by-field comparison would test object identity, not the round-tripped data.
@test all([ position(d1.atoms, :)     == position(d2.atoms, :)     &&
            atomic_number(d1.atoms, :) == atomic_number(d2.atoms, :) &&
            cell_vectors(d1.atoms)    == cell_vectors(d2.atoms)    &&
            periodicity(d1.atoms)     == periodicity(d2.atoms)
            for (d1,d2) in zip(data1,data2)])