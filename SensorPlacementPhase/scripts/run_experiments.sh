### Runs script multiple times, with different parameters. ###

# Create new session, detached.
tmux new-session -d -s WindFarm

# Create the windows for each (and close after script exits).    
#                            Activate 15 CPU threads             Generic Runner Script         Runs       Script To Run                   Solver Args.
# tmux new-window -n:entropy1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        entropy greedy'
# tmux new-window -n:entropy2 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        entropy genetic'
# tmux new-window -n:entropy3 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        entropy mcmc'

# # tmux new-window -n:mutualinfo1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        mutualinfo greedy'
# # tmux new-window -n:mutualinfo2 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        mutualinfo genetic'
# # tmux new-window -n:mutualinfo3 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        mutualinfo mcmc'

# # tmux new-window -n:diffentro1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        diffentro greedy'
# # tmux new-window -n:diffentro2 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        diffentro genetic'
# # tmux new-window -n:diffentro3 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl        diffentro mcmc'

tmux new-window -n:pomcpow01 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow greedy UCB 10'
tmux new-window -n:pomcpow02 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow genetic UCB 10'
tmux new-window -n:pomcpow03 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow mcmc UCB 10'

tmux new-window -n:pomcpow11 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow greedy UCB 25'
tmux new-window -n:pomcpow12 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow genetic UCB 25'
tmux new-window -n:pomcpow13 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow mcmc UCB 25'

tmux new-window -n:pomcpow21 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow greedy UCB 50'
tmux new-window -n:pomcpow22 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow genetic UCB 50'
tmux new-window -n:pomcpow23 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl       pomcpow mcmc UCB 50'



tmux new-window -n:genetic01 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic greedy 3 50 0.8 0.05'
tmux new-window -n:genetic02 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic genetic 3 50 0.8 0.05'
tmux new-window -n:genetic03 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic mcmc 3 50 0.8 0.05'

tmux new-window -n:genetic11 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic greedy 6 100 0.8 0.05'
tmux new-window -n:genetic12 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic genetic 6 100 0.8 0.05'
tmux new-window -n:genetic13 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic mcmc 6 100 0.8 0.05'

tmux new-window -n:genetic21 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic greedy 12 200 0.8 0.05'
tmux new-window -n:genetic22 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic genetic 12 200 0.8 0.05'
tmux new-window -n:genetic23 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl       genetic mcmc 12 200 0.8 0.05'



tmux new-window -n:bayesia01 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian greedy 40 20'
tmux new-window -n:bayesia02 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian genetic 40 20'
tmux new-window -n:bayesia03 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian mcmc 40 20'

tmux new-window -n:bayesia11 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian greedy 200 20'
tmux new-window -n:bayesia12 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian genetic 200 20'
tmux new-window -n:bayesia13 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian mcmc 200 20'

tmux new-window -n:bayesia21 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian greedy 360 20'
tmux new-window -n:bayesia22 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian genetic 360 20'
tmux new-window -n:bayesia23 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl      bayesian mcmc 360 20'