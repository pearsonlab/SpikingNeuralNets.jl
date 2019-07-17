module ReducedTest
using DifferentialEquations, Plots
using SpikingNeuralNets: thresh
using SpikingNeuralNets.SNNPlot: smooth

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
const Iext = 0.35  #0.2346         # external input current
const σnoise = 0.007 # 0.007

# input calculations- input = 0 when t<0
xA(Sa, Sb, I, Inoise, t) = JNaa*Sa - JNab*Sb + Iext + I*thresh(t, 0.0) + Inoise
xB(Sa, Sb, I, Inoise, t) = JNbb*Sb - JNba*Sa + Iext + I*thresh(t, 0.0) + Inoise

# Effective transfer function
a(J) = 239400J + 270
b(J) = 97000J + 108
d(J) = -30J + 0.154
fA(J, x) = J*(-276x + 106) * thresh(x-0.4, 0.0)
function H(J1, J2, x1, x2)
    v = a(J1)*x1 - fA(J2, x2) - b(J1)
    return v / (1 - exp(-d(J1) * v))
end

# 2-variable reduced model
function model(Ia, Ib)
    return function(du, u, p, t, W)
        Sa, Sb = u
        Inoisea, Inoiseb = W
        xa = xA(Sa, Sb, Ia, Inoisea, t)
        xb = xB(Sa, Sb, Ib, Inoiseb, t)

        du[1] = -(Sa/τs) + (1 - Sa) * γ * H(JAaa, JAab, xa, xb)
        du[2] = -(Sb/τs) + (1 - Sb) * γ * H(JAbb, JAba, xb, xa)
    end
end

"""
    run(;c=100, μ0=30.0, S0=[0.0, 0.0], Inoise0=[0.0, 0.0], tstart=-0.5, tstop=1.0, dt=0.1e-3)

Run a reduced 2-variable model of LIP cortical neurons during a random dot motion
task with coherence `c`. `μ0` is the firing rate (in Hz) of input neurons from MT,
`S0` are the initial average NMDA gating variables for the reduced model,
`Inoise0` are the initial noise currents, `tstart` is the  starting time of the
simulation (negative times having no stimulus), `tstop` is the stopping time of
the simulation, and `dt` is the timestep of the simulation.
"""
function run(;c=100, μ0=30.0, S0=[0.0, 0.0], Inoise0=[0.0, 0.0], tstart=-0.5, tstop=1.0, dt=0.1e-3)
    Ia = Jext*μ0*(1 + c/100.0)
    Ib = Jext*μ0*(1 - c/100.0)
    println("Ia: $Ia, Ib: $Ib")

    # noise process: τAMPA dI(t)/dt = -I(t) + W(t)*σnoise*√τAMPA
    W = OrnsteinUhlenbeckProcess(τAMPA, 0.0, σnoise*√τAMPA, tstart, Inoise0)
    prob = RODEProblem(model(Ia, Ib), S0, (tstart, tstop), noise=W)
    println(prob)
    sol = solve(prob, reltol=1e-15, absstol=1e-15, dt=dt, maxiters=1e10)

    # Reconstruct the firing rates
    Sa, Sb = eachrow(reduce(hcat, sol.u))
    Inoisea, Inoiseb = eachrow(reduce(hcat, sol.W.u[1:length(Sa)]))
    xa = xA.(Sa, Sb, Ia, Inoisea, sol.t)
    xb = xB.(Sa, Sb, Ib, Inoiseb, sol.t)
    ra = H.(JAaa, JAab, xa, xb)
    rb = H.(JAbb, JAba, xb, xa)
    #ras, ta = smooth(H.(JAaa, JAab, xa, xb); window=round(Int, 50e-3/dt), dt=round(Int, 5e-3/dt), timestep=dt, tohz=1.0)
    #rbs, tb = smooth(H.(JAbb, JAba, xb, xa); window=round(Int, 50e-3/dt), dt=round(Int, 5e-3/dt), timestep=dt, tohz=1.0)

    display(plot(plot(sol, label=["sA", "sB"]),
                 plot(sol.t, [xa, xb], label=["xA", "xB"]),
                 plot(sol.t, [ra, rb], label=["rA", "rB"]),
                 layout=(3, 1)))

    return (sol.t, Sa, Sb, xa, xb, ra, rb, Inoisea, Inoiseb)

end

end
