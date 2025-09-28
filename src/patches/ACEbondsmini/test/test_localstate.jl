
using ACE, JuLIP, ACEbonds, LinearAlgebra
using ACEbonds: bonds 
using ACE: State 

##

at = bulk(:Si, cubic=true) * 3 
set_pbc!(at, false )
rattle!(at, 0.3)

rbcut = 3.0 
rcut = 5.0 
zcut = 7.0
env_filter = (r, z) -> (r < rcut) && (abs(z) < zcut )

it = bonds(at, rbcut, rbcut/2+zcut, env_filter)

(i, j, rr0, Js, Rs, Zs), state = iterate(it)

for (i1, j1, rr01, Js1, Rs1, Zs1) in bonds(at, rbcut, rbcut/2+zcut, env_filter)
   if i1 == length(at) ÷ 2
      i = i1; j = j1; rr0 = rr01; Js = Js1; Rs = Rs1; Zs = Zs1 
      break 
   end
end

# now we can transform it to an ACE.jl compatible environment 
env = ACEbonds.eucl2cyl(rr0, at.Z[i], at.Z[j], Rs, Zs)


# ## Manual debugging of the bond environments 

# using GLMakie, Makie 

# function plot_bondenv(i, j, Js, at)
#    X = at.X
#    xb = [ X[i][1], X[j][1] ]
#    yb = [ X[i][2], X[j][2] ]
#    zb = [ X[i][3], X[j][3] ]
#    colb = [:red, :red]

#    Xenv = X[Js[2:end]]
#    xe = [ x[1] for x in Xenv ]
#    ye = [ x[2] for x in Xenv ]
#    ze = [ x[3] for x in Xenv ]
#    cole = [:blue for _ in Xenv]

#    Xrest = X[setdiff(1:length(X), union([i,j], Js))]
#    xr = [ x[1] for x in Xrest ]
#    yr = [ x[2] for x in Xrest ]
#    zr = [ x[3] for x in Xrest ]
#    colr = [(:green, 0.3) for _ in Xrest]
   
#    scene = meshscatter([xb; xe; xr], [yb; ye; yr], [zb; ze; zr], 
#                        color = [colb; cole; colr], 
#                        markersize=0.7)
#    lines!(xb, yb, zb, color=:red, linewidth=10 )     
#    return scene                  
# end

# scene = plot_bondenv(i, j, Js, at);
# window = display(scene)
# # close(window) to close it. 