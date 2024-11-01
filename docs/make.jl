# Instructions for building docs locally: 
#   - activate and resolve/up docs Project
#   - ] dev ..  to link to the *current* version of ACEfriction
#   - julia --project=. make.jl  or   julia --project=docs docs/make.jl
#

using Documenter, ACEfriction

# makedocs(sitename="ACEfriction.jl")


makedocs(;
    # modules=[ACEfriction],
    authors="Matthias Sachs <e.matthias.sachs@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEfriction.jl/blob/{commit}{path}#{line}",
    sitename="ACEfriction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEfriction.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => Any[
            "Installation Guide" => "installation.md",
            "Overview" => "overview.md"
            ], 
        "Workflow Examples" => Any[
                "Fitting an Electronic Friction Tensor" => "fitting-eft.md",
                "Fitting a Dissipative Particle Dynamics Friction Model" => "fitting-mbdpd.md"
            ],
        "Function Manual" => Any[
            "function-manual.md",
        ]
    ]
    )


deploydocs(;
    repo="github.com/ACEsuit/ACEfriction.jl.git",
    devbranch="main",
    push_preview=true,
)
