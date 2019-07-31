Code used to generate plot:

#  include("test/WTA.jl")
#  (snn, ) = WTA.run(ones(Int, 10), 10.0; iterations=250)
#  rasterplot(spiketrain(snn, 250))

Parameters: N=10, gamma=10.0, iterations=250
