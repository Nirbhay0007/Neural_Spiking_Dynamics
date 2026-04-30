# optimization 
optf  = OptimizationFunction(total_loss, Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p)
# Use Adam with a learning rate of 0.01
result = solve(optprob, Adam(0.001), maxiters = 100)