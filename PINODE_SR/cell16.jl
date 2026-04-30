function total_loss(p, _)
          # Forward pass ( Data loss part)
          
          
          sol = solve(nn, Heun(),p=θ,reltol=1e-6, abstol=1e-6,sensealg=InterpolatingAdjoint())
          # Data loss
          loss_data = mean(abs2, Array(sol).-z_train)
          loss_phys = pyhysics_loss(nn, z_train, t_train, scale_factors)
          # combine with lambda
          λ = 1.5 
          return loss_data + λ * loss_phys

end