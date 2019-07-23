using Plots, DifferentialEquations
dt = 0.001
σnoise = 0.007
τAMPA = 2e-3
RNG = MersenneTwister(10)
RNG2 = MersenneTwister(10)

function model1(du, u, p, t, W)
    x, I = u
    du[1] = 0.0
    du[2] = (W[1]*σnoise*√τAMPA - I) / τAMPA
end

prob = RODEProblem(model1, [0.0, 0.0], (0.0, 1.0), noise=WienerProcess(0.0, 0.0, rng=RNG, reseed=false))
sol = solve(prob, saveat=dt, dt=dt, reltol=1e-15, abstol=1e-15)
display(plot(sol))

function model2(du, u, p, t, W)
    du[1] = 2*u[1]*sin(W[1])
end

W = OrnsteinUhlenbeckProcess(τAMPA, 0.0, σnoise*√τAMPA, 0.0, [0.0], rng=RNG2, reseed=false)
prob2 = RODEProblem(model2, [0.0], (0.0, 1.0), noise=W)
sol2 = solve(prob2, saveat=dt, dt=dt, reltol=1e-15, abstol=1e-15)
display(plot(sol2.W.t, reduce(vcat, sol2.W.u)))
