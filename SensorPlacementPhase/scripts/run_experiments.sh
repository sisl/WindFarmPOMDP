### Runs script multiple times, with different parameters. ###

# Create new session, detached.
tmux new-session -d -s WindFarm

# Create the windows for each (and close after script exits).    
#                            Activate 15 CPU threads             Generic Runner Script         Runs       Script To Run                   Solver Args.
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        entropy greedy'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        entropy genetic'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        entropy mcmc'

tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        mutualinfo greedy'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        mutualinfo genetic'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        mutualinfo mcmc'

tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        diffentro greedy'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        diffentro genetic'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_greedy.jl        diffentro mcmc'

tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_pomcpow.jl       pomcpow greedy UCB'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_pomcpow.jl       pomcpow genetic UCB'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_pomcpow.jl       pomcpow mcmc UCB'

tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_genetic.jl       genetic greedy'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_genetic.jl       genetic genetic'
tmux new-window -n:mywindow 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        3        solve_SensorPP_genetic.jl       genetic mcmc'
