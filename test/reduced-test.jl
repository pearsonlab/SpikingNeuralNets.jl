module ReducedTest
using DifferentialEquations, Distributions, Plots
using SpikingNeuralNets: thresh

# Fixed Parameters
const Jext = 0.2243e-3                # weights
const JNaa = const JNbb = 0.1561
const JNab = const JNba = 0.0264
const JAaa = const JAbb = 9.9026e-4
const JAab = const JAba = 6.5177e-5

const τs = 100e-3           # time constants (s)
const τAMPA = 2e-3
const θ = 15.0              # firing rate threshold (Hz)
const γ = 0.641
const Iext = 0.2346         # external input current
const η = Normal()
const σnoise = 0.007

# input calculations
xA(Sa, Sb, Ia, Inoise, t) = JNaa*Sa - JNab*Sb + Iext + Ia*thresh(t, 0.0) + Inoise
xB(Sa, Sb, Ib, Inoise, t) = JNbb*Sb - JNba*Sa + Iext + Ib*thresh(t, 0.0) + Inoise

# Effective transfer function
a(J) = 239400J + 270
b(J) = 97000J + 108
d(J) = -30J + 0.154
fA(J, x) = J*(-276x + 106) * thresh(x-0.4, 0.0)
function H(J1, J2, x1, x2)
    v = a(J1)*x1 - fA(J2, x2) - b(J1)
    return v / (1 - exp(-d(J1) * v))
end

function model(Ia, Ib)
    return function(du, u, p, t)
        Sa, Inoisea, Sb, Inoiseb = u
        xa = xA(Sa, Sb, Ia, Inoisea, t)
        xb = xB(Sa, Sb, Ib, Inoiseb, t)
        #println("Sa: $Sa, Sb: $Sb, Inoise: $Inoise")

        du[1] = -(Sa/τs) + (1 - Sa) * γ * H(JAaa, JAab, xa, xb)
        du[2] = (rand(η)*σnoise*sqrt(τAMPA) - Inoisea) / τAMPA
        du[3] = -(Sb/τs) + (1 - Sb) * γ * H(JAbb, JAba, xb, xa)
        du[4] = (rand(η)*σnoise*sqrt(τAMPA) - Inoiseb) / τAMPA

        #println("dSa: $(du[1]), dSb: $(du[2]), dInoise: $(du[3])")
    end
end

function run(;c=100, μ0=30.0, sA0=0.0, sB0=0.0, Inoisea0=0.0, Inoiseb0=0.0, tstart=-0.5, tstop=1.0, step=0.01e-3)
    Ia = Jext*μ0*(1 + c/100.0)
    Ib = Jext*μ0*(1 - c/100.0)
    println("Ia: $Ia, Ib: $Ib")

    prob = ODEProblem(model(Ia, Ib), [sA0, sB0, Inoisea0, Inoiseb0], (tstart, tstop))
    println(prob)
    sol = solve(prob, reltol=1e-15, absstol=1e-15, saveat=step, maxiters=1e10)
    display(plot(sol))

    Sa, Inoisea, Sb, Inoiseb = eachrow(reduce(hcat, sol.u))
    xa = xA.(Sa, Sb, Ia, Inoisea, sol.t)
    xb = xB.(Sa, Sb, Ib, Inoiseb, sol.t)
    ra = H.(JAaa, JAab, xa, xb)
    rb = H.(JAbb, JAba, xb, xa)

    return (sol.t, Sa, Sb, xa, xb, ra, rb, Inoisea, Inoiseb)
end

end
