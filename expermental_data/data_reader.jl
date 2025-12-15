using DataFrames
using CSV
using Plots
using JLD

df = CSV.read("expermental_data/neuron_data_noise1.csv", DataFrame)



# WE are going to find the first spike V above -20mV
spike_index = findfirst(df.voltage_mv.> -20)
plot(df.time_ms,df.voltage_mv)