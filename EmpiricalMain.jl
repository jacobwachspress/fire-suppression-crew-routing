include("BranchAndPrice.jl")

using JuMP, Gurobi, JSON, Profile, ArgParse, Logging, IterTools

const GRB_ENV = Gurobi.Env()


function get_command_line_args()
	arg_parse_settings = ArgParseSettings()
	@add_arg_table arg_parse_settings begin
		"--debug"
		help = "run in debug mode, exposing all logging that uses @debug macro"
		action = :store_true
	end
	return parse_args(arg_parse_settings)
end


args = get_command_line_args()

io = open("logs_precompile_5.txt", "w")
if args["debug"] == true
	global_logger(ConsoleLogger(io, Logging.Debug, show_limited = false))
else
	global_logger(ConsoleLogger(io, Logging.Info, show_limited = false))
end

# precompile
io = open("logs_60.txt", "w")
if args["debug"] == true
	global_logger(ConsoleLogger(io, Logging.Debug, show_limited = false))
else
	global_logger(ConsoleLogger(io, Logging.Info, show_limited = false))
end
num_fires = 14	
num_crews = 12
num_time_periods = 14
travel_speed = 40.0 * 6.0
GC.gc()

crew_routes, fire_plans, crew_models, fire_models, cut_data = initialize_data_structures(num_fires, num_crews, num_time_periods, 20, travel_speed, from_empirical = true)
for j in 1:num_crews
	no_fire_anticipation!(crew_models[j], [fsp.start_time_period for fsp in fire_models])
end

# add a dummy plan with cost 0 and no crew demands
for fire in 1:num_fires
	@info "adding dummy plan for fire" fire
	new_plan_ix =
		add_column_to_plan_data!(fire_plans, fire, 0.0, zeros(Int64, num_time_periods), Int[])
	if new_plan_ix == -1
		@error "failed to add dummy plan for fire" fire
		error()
	end
	@info "added dummy plan for fire" fire "with index" new_plan_ix
end

for t in 0:14

	global crew_routes, fire_plans, crew_models, fire_models, cut_data

	crew_routes = CrewRouteData(Int(floor(6 * 1e6 / num_crews)), num_fires, num_crews, num_time_periods)
	fire_plans = FirePlanData(Int(floor(6 * 1e6  / num_crews)), num_fires, num_time_periods)
	cut_data = CutData(num_crews, num_fires, num_time_periods)

	result = branch_and_price(num_fires,
		num_crews,
		num_time_periods,
		current_time = t,
		from_empirical = true, 
		travel_speed = travel_speed,
		crew_routes = crew_routes,
		fire_plans = fire_plans,
		crew_models = crew_models,
		fire_models = fire_models,
		cut_data = cut_data
		)
		# Unpack as many variables as branch_and_price returns, e.g.:
	explored_nodes, ubs, lbs, columns, heuristic_times, times, time_1, root_node_ip_sol, root_node_ip_sol_time, fire_arcs_used, crew_arcs_used = result
	@info "final arcs used" fire_arcs_used, crew_arcs_used

	for g in 1:num_fires
		@info "before modify_in_arcs_and_out_arcs!" fire_models[g].state_in_arcs fire_models[g].state_out_arcs fire_arcs_used[g]
		if !isnothing(fire_models[g].start_time_period) && fire_models[g].start_time_period > t
			@info "fire model start time period is greater than current time, skipping modify_in_arcs_and_out_arcs!" g
			continue
		end
		modify_in_arcs_and_out_arcs!(fire_models[g], t+1, fire_arcs_used[g], FM.TIME_FROM)
		@info "after modify_in_arcs_and_out_arcs!" fire_models[g].state_in_arcs fire_models[g].state_out_arcs
	end
	for j in 1:num_crews
		@info "before modify_in_arcs_and_out_arcs!" crew_models[j].state_in_arcs crew_models[j].state_out_arcs crew_arcs_used[j]
		modify_in_arcs_and_out_arcs!(crew_models[j], t+1, crew_arcs_used[j], CM.TIME_FROM)
		@info "after modify_in_arcs_and_out_arcs!" crew_models[j].state_in_arcs crew_models[j].state_out_arcs
	end

	# now extract the arc data and costs from the fire_arcs_used and crew_arcs_used and the models
	fire_arcs = Vector{Matrix{Int64}}(undef, num_fires)
	fire_arc_costs = Vector{Vector{Float64}}(undef, num_fires)
	crew_arcs = Vector{Matrix{Int64}}(undef, num_crews)
	crew_arc_costs = Vector{Vector{Float64}}(undef, num_crews)
	for g in 1:num_fires
		fire_arcs[g] = fire_models[g].wide_arcs[:, reverse(fire_arcs_used[g])]
		fire_arc_costs[g] = fire_models[g].arc_costs[reverse(fire_arcs_used[g])]
	end
	for j in 1:num_crews
		crew_arcs[j] = crew_models[j].wide_arcs[:, reverse(crew_arcs_used[j])]
		crew_arc_costs[j] = crew_models[j].arc_costs[reverse(crew_arcs_used[j])]
	end

	# write these to files
	for g in 1:num_fires
		open("data/output/fire_arcs_$(g)_$(t).json", "w") do io
			JSON.print(io, fire_arcs[g])
		end
		open("data/output/fire_arc_costs_$(g)_$(t).json", "w") do io
			JSON.print(io, fire_arc_costs[g])
		end
	end
	for j in 1:num_crews
		open("data/output/crew_arcs_$(j)_$(t).json", "w") do io
			JSON.print(io, crew_arcs[j])
		end
		open("data/output/crew_arc_costs_$(j)_$(t).json", "w") do io
			JSON.print(io, crew_arc_costs[j])
		end
	end
end

# io = open("logs_40.txt", "w")
# if args["debug"] == true
# 	global_logger(ConsoleLogger(io, Logging.Debug, show_limited = false))
# else
# 	global_logger(ConsoleLogger(io, Logging.Info, show_limited = false))
# end
# branch_and_price(8, 40, 14, from_empirical = true, travel_speed = 40.0 * 16.0)
# io = open("logs_60.txt", "w")
# if args["debug"] == true
# 	global_logger(ConsoleLogger(io, Logging.Debug, show_limited = false))
# else
# 	global_logger(ConsoleLogger(io, Logging.Info, show_limited = false))
# end
# branch_and_price(8, 60, 14, from_empirical = true, travel_speed = 40.0 * 16.0)
