include("CommonStructs.jl")
include("BranchingRules.jl")
include("CuttingPlanes.jl")

using Gurobi, Statistics, JSON, NPZ

mutable struct BranchAndBoundNode

	const ix::Int64
	const parent::Union{Nothing, BranchAndBoundNode}
	const new_crew_branching_rules::Vector{CrewAssignmentBranchingRule}
	const new_fire_branching_rules::Vector{FireDemandBranchingRule}
	const new_global_fire_allotment_branching_rules::Vector{
		GlobalFireAllotmentBranchingRule,
	}
	const cut_data::CutData
	children::Vector{BranchAndBoundNode}
	l_bound::Float64
	u_bound::Float64
	master_problem::Union{Nothing, RestrictedMasterProblem}
	heuristic_found_master_problem::Union{Nothing, RestrictedMasterProblem}
	feasible::Union{Nothing, Bool}
end

function BranchAndBoundNode(
	;
	ix::Int64,
	parent::Union{Nothing, BranchAndBoundNode},
	cut_data::CutData,
	new_crew_branching_rules::Vector{CrewAssignmentBranchingRule} = CrewAssignmentBranchingRule[],
	new_fire_branching_rules::Vector{FireDemandBranchingRule} = FireDemandBranchingRule[],
	new_global_fire_allotment_branching_rules::Vector{
		GlobalFireAllotmentBranchingRule,
	} = GlobalFireAllotmentBranchingRule[],
	children::Vector{BranchAndBoundNode} = BranchAndBoundNode[],
	l_bound::Float64 = -Inf,
	u_bound::Float64 = Inf,
	master_problem::Union{Nothing, RestrictedMasterProblem} = nothing,
	heuristic_found_master_problem::Union{Nothing, RestrictedMasterProblem} = nothing,
	feasible::Union{Nothing, Bool} = nothing)

	return BranchAndBoundNode(
		ix,
		parent,
		new_crew_branching_rules,
		new_fire_branching_rules,
		new_global_fire_allotment_branching_rules,
		cut_data,
		children,
		l_bound,
		u_bound,
		master_problem,
		heuristic_found_master_problem,
		feasible,
	)
end

###############

"""
Perform price-and-cut algorithm to solve the (relaxed) restricted master problem `rmp`.
This algorithm alternates between column generation and cut generation phases and returns
rmp with additional columns, cuts, and a solution that certifies a lower bound to the
integer problem (or an infeasibility certificate).

# Arguments

  - `rmp::RestrictedMasterProblem` (modified): The restricted master problem instance representing the optimization problem.
  - `crew_routes::CrewRouteData` (modified): Data structure containing information (cost, assignments) about all generated crew routes.
  - `fire_plans::FirePlanData` (modified): Data structure containing information (cost, demands) about all generated fire plans.
  - `cut_data::CutData` (modified): Data structure containing information about cuts, including lookups to incorporate coeffs into master problem and subproblems.
  - `crew_subproblems::Vector{TimeSpaceNetwork}`: Vector of data structures containing all static info about crew subproblems.
  - `fire_subproblems::Vector{TimeSpaceNetwork}`: Vector of data structures containing all static info about fire subproblems.
  - cut_search_enumeration_limit::{Int64}: Enumeration limit for cut generation search, which can theoretically grow exponentially large
  - `crew_rules::Vector{CrewAssignmentBranchingRule}`: Vector of active branching rules that indicate whether crew j suppresses fire g at time t.
  - `fire_rules::Vector{FireDemandBranchingRule}`: Vector of active branching rules that indicate whether fire g demands <=d crews or >d crews at time t.
  - `global_fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule}`: Vector of active branching rules that may place a cap on demand across all fires and times
  - TODO finish this
"""
function price_and_cut!!!!(
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	cut_data::CutData,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	cut_search_enumeration_limit::Int64,
	crew_rules::Vector{CrewAssignmentBranchingRule},
	fire_rules::Vector{FireDemandBranchingRule},
	global_fire_allotment_rules::Vector{GlobalFireAllotmentBranchingRule},
	fires_to_ignore::Vector{Int64};
	gub_cover_cuts::Bool,
	general_gub_cuts::String,
	single_fire_cuts::Bool,
	decrease_gub_allots::Bool,
	single_fire_lift::Bool,
	soft_time_limit::Float64,
	loop_max::Int,
	relative_improvement_cut_req::Float64,
	upper_bound::Float64 = 1e10,
	log_progress_file = nothing,
	time_limit = Inf,
)

	log_flag = ~isnothing(log_progress_file)

	if log_flag
		ts = []
		dcg_logs = []
		objs = []
		num_routes = []
		num_plans = []
		num_cuts_found = []
		active_cuts = []
		cut_times = []
	end

	t = time()
	loop_ix = 0
	most_recent_obj = 0

	while true

		# run DCG, adding columns as needed
		dcg_times = double_column_generation!!!!(
			rmp,
			crew_routes,
			fire_plans,
			cut_data,
			crew_subproblems,
			fire_subproblems,
			crew_rules,
			fire_rules,
			global_fire_allotment_rules,
			fires_to_ignore,
			timing = log_flag,
			upper_bound = upper_bound,
			time_limit = time_limit,
		)
		if (rmp.termination_status == MOI.OBJECTIVE_LIMIT) || (rmp.termination_status == MOI.INFEASIBLE)
			@debug "no more cuts needed"
			break
		end

		current_obj = objective_value(rmp.model)
		if log_flag
			push!(ts, time() - t)
			push!(dcg_logs, dcg_times)
			push!(objs, current_obj)
			push!(num_routes, sum(crew_routes.routes_per_crew))
			push!(num_plans, sum(fire_plans.plans_per_fire))
			push!(num_cuts_found, sum(cut_data.cuts_per_time))
			push!(
				active_cuts,
				sum([
					1 for i in eachindex(rmp.gub_cover_cuts) if
					dual(rmp.gub_cover_cuts[i]) > 1e-4
				]),
			)
		end
		loop_ix += 1

		if loop_ix > loop_max
			@info "halting cut generation early, exceeded max loops" loop_ix loop_max
			break
		end

		if most_recent_obj / current_obj > 1 - relative_improvement_cut_req
			@info "halting cut generation early, too small improvement" loop_ix
			break
		end
		most_recent_obj = current_obj

		if time() - t > soft_time_limit
			@info "halting cut generation early, reached cut time limit" soft_time_limit
			break
		end

		# add cuts
		t2 = @elapsed num_cuts = find_and_incorporate_knapsack_gub_cuts!!(
			cut_data,
			cut_search_enumeration_limit,
			rmp,
			crew_routes,
			fire_plans,
			crew_subproblems,
			fire_subproblems,
			gub_cover_cuts = gub_cover_cuts,
			general_gub_cuts = general_gub_cuts,
			single_fire_cuts = single_fire_cuts,
			decrease_gub_allots = decrease_gub_allots,
			single_fire_lift = single_fire_lift,
		)

		if log_flag
			push!(cut_times, t2)
		end

		@debug "after cuts" num_cuts objective_value(rmp.model)
		if (num_cuts == 0)
			@debug "No cuts found, breaking" loop_ix
			break
		end

		if num_cuts == 0
			@debug "No cuts found, breaking" loop_ix
			break

		end
	end

	if log_flag
		outputs = Dict(
			"times" => ts,
			"dcg_times" => dcg_logs,
			"objectives" => objs,
			"num_routes" => num_routes,
			"num_plans" => num_plans,
			"num_cuts" => num_cuts_found,
			"num_active_cuts" => active_cuts,
			"cut_times" => cut_times,
		)
		open(log_progress_file, "w") do f
			JSON.print(f, outputs, 4)
		end
	end

	@debug "After price-and-cut" dual.(rmp.supply_demand_linking)

end

function branch_and_price(
	num_fires::Int,
	num_crews::Int,
	num_time_periods::Int;
	current_time = 0,
	from_empirical = false,
	line_per_crew = 20,
	travel_speed = 640.0,
	max_nodes = 10000,
	algo_tracking = false,
	branching_strategy = "linking_dual_max_variance",
	cut_search_enumeration_limit = 10000,
	cut_loop_max = 10,
	price_and_cut_soft_time_limit = 180.0,
	relative_improvement_cut_req = 1e-10,
	soft_heuristic_time_limit = 60.0,
	hard_heuristic_iteration_limit = 3,
	heuristic_must_improve_rounds = 2,
	heuristic_cadence_secs = 120.0,
	total_time_limit = 1800.0,
	bb_node_gub_cover_cuts = false,
	bb_node_general_gub_cuts = "adaptive",
	bb_node_single_fire_cuts = false,
	bb_node_decrease_gub_allots = false,
	bb_node_single_fire_lift = false,
	heuristic_gub_cover_cuts = false,
	heuristic_general_gub_cuts = "adaptive",
	heuristic_single_fire_cuts = false,
	heuristic_decrease_gub_allots = false,
	heuristic_single_fire_lift = false,
	root_node_ip = false,
	price_and_cut_file = nothing,
	crew_routes = nothing,
	fire_plans = nothing, 
	crew_models  = nothing,
	fire_models = nothing,
	cut_data  = nothing,
)
	start_time = time()
	root_node_ip_sol = 0.0
	root_node_ip_sol_time = 0.0

	if crew_routes === nothing
		@info "Initializing data structures"
		# initialize input data
		@time crew_routes, fire_plans, crew_models, fire_models, cut_data =
			initialize_data_structures(num_fires, num_crews, num_time_periods, line_per_crew, travel_speed, from_empirical = from_empirical)
		GC.gc()
		algo_tracking ?
		(@info "Checkpoint after initializing data structures" time() - start_time) :
		nothing
	end

	fires_to_ignore = Int64[]
	for fire in 1:num_fires
		if !isnothing(fire_models[fire].start_time_period) && fire_models[fire].start_time_period > current_time
			push!(fires_to_ignore, fire)
			@info "Ignoring fire" fire "because it starts at time" fire_models[fire].start_time_period
		end
	end

	explored_nodes = []
	ubs = []
	lbs = []
	columns = []
	heuristic_times = []
	times = []
	time_1 = time() - start_time

	# initialize last heuristic time to ensure running heuristic at root node
	last_heuristic_time = time_1 - heuristic_cadence_secs 

	# initialize nodes list with the root node
	nodes = BranchAndBoundNode[]
	first_node = BranchAndBoundNode(ix = 1, parent = nothing, cut_data = cut_data)
	push!(nodes, first_node)

	# initialize global variables to track in branch-and-bound tree
	ub = Inf
	lb = 0
	ub_ix::Int = -1
	routes_best_sol = nothing
	plans_best_sol = nothing
	fire_arcs_used = nothing
	crew_arcs_used = nothing

	## breadth-first search for now, can get smarter/add options

	unexplored = [i for i in eachindex(nodes) if isnothing(nodes[i].master_problem)]
	node_lbs = []
	for i in unexplored
		if (~isnothing(nodes[i].parent))
			push!(node_lbs, nodes[i].parent.l_bound)
		else
			@warn "Node with no parent" i
			push!(node_lbs, 0)
		end
	end

	node_explored_count = 0
	# while there are more nodes to explore
	while length(unexplored) > 0

		node_explored_count += 1

		### breadth first ###
		# node_ix = unexplored[1]

		### raise lb ###
		min_lb = findmin(node_lbs)[1]
		node_ix = unexplored[findmin(node_lbs)[2]]

		# break early if we are close enough to solved
		if min_lb / ub > 1 - 1e-8
			@info "Solved to tolerance" min_lb ub
			break
		end

		# else explore the next node
		explore_node!!(
			nodes[node_ix],
			nodes,
			ub,
			crew_routes,
			fire_plans,
			crew_models,
			fire_models,
			cut_search_enumeration_limit,
			nothing,
			GRB_ENV,
			fires_to_ignore = fires_to_ignore,
			price_and_cut_soft_time_limit = price_and_cut_soft_time_limit,
			cut_loop_max = cut_loop_max,
			relative_improvement_cut_req = relative_improvement_cut_req,
			gub_cover_cuts = bb_node_gub_cover_cuts,
			general_gub_cuts = bb_node_general_gub_cuts,
			single_fire_cuts = bb_node_single_fire_cuts,
			decrease_gub_allots = bb_node_decrease_gub_allots,
			single_fire_lift = bb_node_single_fire_lift,
			log_cuts_file = price_and_cut_file,
			branching_strategy = branching_strategy,
		)

		if time() - start_time > total_time_limit
			@info "Full time limit reached"
			break
		end

		heuristic_time = 0.0
		if (heuristic_cadence_secs < time() - last_heuristic_time) &&
		   (soft_heuristic_time_limit > 0.0) && (lb / ub < 1 - 1e-3) # gurobi has 1e-4 tol I can't fix
		   last_heuristic_time = time()	
		   heuristic_ub, ub_rmp, routes_best_sol, plans_best_sol =
				heuristic_upper_bound!!(
					crew_routes,
					fire_plans,
					nodes[node_ix],
					hard_heuristic_iteration_limit,
					soft_heuristic_time_limit,
					heuristic_must_improve_rounds,
					crew_models,
					fire_models,
					cut_search_enumeration_limit,
					GRB_ENV,
					fires_to_ignore = fires_to_ignore,
					routes_best_sol = routes_best_sol,
					plans_best_sol = plans_best_sol,
					price_and_cut_soft_time_limit = price_and_cut_soft_time_limit,
					cut_loop_max = cut_loop_max,
					relative_improvement_cut_req = relative_improvement_cut_req,
					gub_cover_cuts = heuristic_gub_cover_cuts,
					general_gub_cuts = heuristic_general_gub_cuts,
					single_fire_cuts = heuristic_single_fire_cuts,
					decrease_gub_allots = heuristic_decrease_gub_allots,
					single_fire_lift = heuristic_single_fire_lift,
				)

			if time() - start_time > total_time_limit
				@info "Full time limit reached"
				break
			end

			heuristic_time = time() - last_heuristic_time

			nodes[node_ix].heuristic_found_master_problem = ub_rmp

			if heuristic_ub < nodes[node_ix].u_bound
				# nodes[node_ix].u_bound = heuristic_ub

				# fire_arcs_used, crew_arcs_used = get_fire_and_crew_arcs_used(ub_rmp,
				# 	crew_routes,
				# 	fire_plans,
				# )
			end
		end

		## now that we have done the heuristic, can mess up rmp and lose duals
		if root_node_ip && (node_explored_count == 1)

			root_node_ip_sol_time = @elapsed root_node_ip_sol, obj_bound, _, _ =
				find_integer_solution(
					nodes[node_ix].master_problem,
					Inf,
					1200000,
					4000000,
					1200.0,)			
			@info "restore_integrality" t obj obj_bound
		end



		# if this node has an integer solution, check if we have found 
		# a better solution than the incumbent
		# TODO keep track if it comes from heuristic or no
		if nodes[node_ix].u_bound < ub

			fire_allots, crew_allots = get_fire_and_crew_incumbent_weighted_average(nodes[node_ix].master_problem,
				crew_routes,
				fire_plans,
			)
			fire_cost, crew_cost = get_cost_due_to_fires_and_crews(nodes[node_ix].master_problem,
				crew_routes,
				fire_plans,
			)
			fire_arcs_used, crew_arcs_used = get_fire_and_crew_arcs_used(nodes[node_ix].master_problem,
				crew_routes,
				fire_plans,
			)

			# if it is not a directory, make the directory data/output
			isdir("data/output") || mkdir("data/output")

			# extract the arc data from the subproblems
			for fire in 1:num_fires
				arcs_used = fire_arcs_used[fire]
				arcs_used = reverse(arcs_used)
				restricted_long_arcs = fire_models[fire].long_arcs[arcs_used, :]
				restricted_costs = fire_models[fire].arc_costs[arcs_used]
				# remove the first column, which is extraneous
				restricted_long_arcs = restricted_long_arcs[:, 2:end]
				NPZ.npzwrite("data/output/fire_$(fire)_arcs.npy", restricted_long_arcs)
				NPZ.npzwrite("data/output/fire_$(fire)_costs.npy", restricted_costs)
			end

			for crew in 1:num_crews
				arcs_used = crew_arcs_used[crew]
				arcs_used = reverse(arcs_used)
				restricted_long_arcs = crew_models[crew].long_arcs[arcs_used, :]
				# remove the first column, which is extraneous
				restricted_long_arcs = restricted_long_arcs[:, 2:end]
				restricted_costs = crew_models[crew].arc_costs[arcs_used]
				NPZ.npzwrite("data/output/crew_$(crew)_arcs.npy", restricted_long_arcs)
				NPZ.npzwrite("data/output/crew_$(crew)_costs.npy", restricted_costs)
			end

			@info "new incumbent" fire_allots crew_allots fire_cost crew_cost fire_arcs_used crew_arcs_used


			ub = nodes[node_ix].u_bound
			ub_ix = node_ix
		end

		# calculate the best current lower bound by considering all nodes with
		# fully explored children 
		lb = find_lower_bound(nodes[1])

		# print progress
		@info "current bounds" node_ix lb ub
		println(lb)
		println(ub)
		# go to the next node
		@info "number of nodes" node_explored_count length(nodes)
		@info "columns" sum(crew_routes.routes_per_crew) sum(
			fire_plans.plans_per_fire,
		)
		algo_tracking ?
		(@info "Time check" time() - start_time) :
		nothing

		if algo_tracking
			push!(explored_nodes, node_explored_count)
			push!(lbs, lb)
			push!(ubs, ub)
			push!(
				columns,
				sum(crew_routes.routes_per_crew) + sum(fire_plans.plans_per_fire),
			)
			push!(heuristic_times, heuristic_time)
			push!(times, time() - start_time)
		end

		unexplored =
			[i for i in eachindex(nodes) if isnothing(nodes[i].master_problem)]
		node_lbs = []
		for i in unexplored
			if (~isnothing(nodes[i].parent))
				push!(node_lbs, nodes[i].parent.l_bound)
			else
				@warn "Node with no parent" i
				push!(node_lbs, -Inf)
			end
		end

		if node_explored_count >= max_nodes
			@info "Hit node limit, breaking" node_explored_count
			break
		end
	end

	return explored_nodes, ubs, lbs, columns, heuristic_times, times, time_1, root_node_ip_sol, root_node_ip_sol_time, fire_arcs_used, crew_arcs_used

end

function initialize_data_structures(
	num_fires::Int64,
	num_crews::Int64,
	num_time_periods::Int64,
	line_per_crew::Int64,
	travel_speed::Float64;
	from_empirical = false
)
	if !from_empirical
		crew_models = build_crew_models(
			"data/raw/big_fire",
			num_fires,
			num_crews,
			num_time_periods,
			travel_speed,
		)

		fire_models = build_fire_models(
			"data/raw/big_fire",
			num_fires,
			num_crews,
			num_time_periods,
			line_per_crew
		)
	else
		crew_models = build_crew_models_from_empirical(
			num_crews, num_fires, num_time_periods, travel_speed
		)
		fire_models = build_fire_models_from_empirical(
			num_fires, num_crews, num_time_periods
		)
	end


	crew_routes = CrewRouteData(Int(floor(6 * 1e6 / num_crews)), num_fires, num_crews, num_time_periods)
	fire_plans = FirePlanData(Int(floor(6 * 1e6  / num_crews)), num_fires, num_time_periods)
	cut_data = CutData(num_crews, num_fires, num_time_periods)

	return crew_routes, fire_plans, crew_models, fire_models, cut_data
end

# TODO if this is a bottleneck, can cache lower bounds
# in a new field in branch and bound node
function find_lower_bound(node::BranchAndBoundNode)

	# child.l_bound = -Inf if unexplored
	child_lbs = [find_lower_bound(child) for child in node.children]

	if length(child_lbs) > 0
		return max(minimum(child_lbs), node.l_bound)
	else
		return node.l_bound
	end

end


function most_fractional_natural_variable(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	route_values::JuMP.Containers.SparseAxisArray,
	plan_values::JuMP.Containers.SparseAxisArray,
	linking_duals::Union{Nothing, Matrix{Float64}},
)
	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	# calculate variance of B_{gt}, demand at fire g at time t
	fire_means = zeros(Float64, (num_fires, num_time_periods))
	for ix ∈ eachindex(plan_values)
		fire = ix[1]
		plan = ix[2]
		fire_means[fire, :] +=
			plan_values[ix] * fire_plans.crews_present[fire, plan, :]
	end

	fire_fractionals = -((fire_means .- floor.(fire_means) .- 0.5) .^ 2)
	fire_biggest_frac, fire_most_fractional = findmax(fire_fractionals)

	# if some fire has fractional plans, return the info needed to create the branching rule
	if fire_biggest_frac > 1e-10
		return "fire",
		fire_most_fractional,
		fire_biggest_frac,
		fire_means[fire_max_ix]
	end

	# if we reach this point, all fire plans are {0,1}, check crews now

	# calculate variance of A_{cgt}, crew c suppressing fire g at time t
	crew_means = zeros(Float64, (num_crews, num_fires, num_time_periods))
	for ix ∈ eachindex(route_values)
		crew = ix[1]
		route = ix[2]
		crew_means[crew, :, :] +=
			route_values[ix] * crew_routes.fires_fought[crew, route, :, :]
	end

	crew_fractionals = -((crew_means .- floor.(crew_means) .- 0.5) .^ 2)
	crew_biggest_frac, crew_most_fractional = findmax(crew_fractionals)

	# if some crew has fractional plans, return the info needed to create the branching rule
	if crew_biggest_frac > 1e-10
		return "crew",
		crew_most_fractional,
		crew_biggest_frac,
		crew_means[crew_max_ix]
	end

	# since this is not guaranteed to branch if fractional fire plans add to an integer, need backup plan
	return max_variance_natural_variable(
		crew_routes,
		fire_plans,
		route_values,
		plan_values,
		linking_duals,
	)
end


function max_variance_natural_variable(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	route_values::JuMP.Containers.SparseAxisArray,
	plan_values::JuMP.Containers.SparseAxisArray,
	linking_duals::Union{Nothing, Matrix{Float64}},
)
	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	# calculate variance of B_{gt}, demand at fire g at time t
	fire_means = zeros(Float64, (num_fires, num_time_periods))
	fire_sq_means = zeros(Float64, (num_fires, num_time_periods))
	for ix ∈ eachindex(plan_values)
		fire = ix[1]
		plan = ix[2]
		fire_means[fire, :] +=
			plan_values[ix] * fire_plans.crews_present[fire, plan, :]
		fire_sq_means[fire, :] +=
			plan_values[ix] * (fire_plans.crews_present[fire, plan, :] .^ 2)
	end
	fire_variances = fire_sq_means - (fire_means .^ 2)

	# dual-adjust variances, increasing focus on high-leverage fire/times
	if ~isnothing(linking_duals)

		for g ∈ 1:num_fires
			for t ∈ 1:num_time_periods
				fire_variances[g, t] *= ((linking_duals[g, t])^2)
			end
		end
	end

	@debug "Variances" fire_variances # crew_variances

	# get the max fire variance 
	fire_max_var, fire_max_ix = findmax(fire_variances)

	# if some fire has fractional plans, return the info needed to create the branching rule
	if fire_max_var > 0
		return "fire", fire_max_ix, fire_max_var, fire_means[fire_max_ix]
	end

	# if we reach this point, all fire plans are {0,1}, check crews now

	# calculate variance of A_{cgt}, crew c suppressing fire g at time t
	crew_means = zeros(Float64, (num_crews, num_fires, num_time_periods))
	for ix ∈ eachindex(route_values)
		crew = ix[1]
		route = ix[2]
		crew_means[crew, :, :] +=
			route_values[ix] * crew_routes.fires_fought[crew, route, :, :]
	end
	crew_variances = crew_means .* (1 .- crew_means)

	# dual-adjust crew variances
	if ~isnothing(linking_duals)
		for j ∈ 1:num_crews
			for g ∈ 1:num_fires
				for t ∈ 1:num_time_periods
					crew_variances[j, g, t] *= ((linking_duals[g, t])^2)
				end
			end
		end
	end
	# get the max variance for crews
	crew_max_var, crew_max_ix = findmax(crew_variances)

	# return the info needed to create the branching rule
	# (could be zeros if crews are intege; this should be eliminated earlier)
	return "crew", crew_max_ix, crew_max_var, crew_means[crew_max_ix]

end


function apply_branching_rule(
	crew_avail_ixs::Vector{Vector{Int64}},
	crew_routes::CrewRouteData,
	branching_rule::CrewAssignmentBranchingRule)

	output = [Int64[] for fire in 1:size(crew_avail_ixs)[1]]
	for crew in 1:size(crew_avail_ixs)[1]
		ixs = copy(crew_avail_ixs[crew])
		if branching_rule.crew_ix == crew
			ixs = [
				i for i in ixs if satisfies_branching_rule(
					branching_rule,
					crew_routes.fires_fought[branching_rule.crew_ix, i, :, :],
				)
			]
		end
		output[crew] = ixs
		ixs = copy(crew_avail_ixs[crew])
		if branching_rule.crew_ix == crew
			ixs = [
				i for i in ixs if satisfies_branching_rule(
					branching_rule,
					crew_routes.fires_fought[branching_rule.crew_ix, i, :, :],
				)
			]
		end
		output[crew] = ixs
	end

	return output
end

function apply_branching_rule(
	fire_avail_ixs::Vector{Vector{Int64}},
	fire_plans::FirePlanData,
	branching_rule::FireDemandBranchingRule,
)

	output = [Int64[] for fire in 1:size(fire_avail_ixs)[1]]
	for fire in 1:size(fire_avail_ixs)[1]
		ixs = copy(fire_avail_ixs[fire])
		if branching_rule.fire_ix == fire
			ixs = [
				i for i in ixs if satisfies_branching_rule(
					branching_rule,
					fire_plans.crews_present[branching_rule.fire_ix, i, :],
				)
			]
		end
		output[fire] = ixs
		ixs = copy(fire_avail_ixs[fire])
		if branching_rule.fire_ix == fire
			ixs = [
				i for i in ixs if satisfies_branching_rule(
					branching_rule,
					fire_plans.crews_present[branching_rule.fire_ix, i, :],
				)
			]
		end
		output[fire] = ixs
	end


	return output

end

function find_integer_solution(
	solved_rmp::RestrictedMasterProblem,
	upper_bound::Float64,
	fire_column_limit::Int64,
	crew_column_limit::Int64,
	time_limit::Union{Int, Float64};
	warm_start_routes = nothing,
	warm_start_plans = nothing,
)
	# get reduced costs
	rc_plans = reduced_cost.(solved_rmp.plans)
	rc_routes = reduced_cost.(solved_rmp.routes)

	# get relaxed objective value
	lp_objective = objective_value(solved_rmp.model)

	if fire_column_limit > length(rc_plans)
		@debug "fewer fire plans than limit" length(rc_plans) fire_column_limit
		fire_column_limit = length(rc_plans)
	end

	if crew_column_limit > length(rc_routes)
		@debug "fewer crew routes than limit" length(rc_routes) crew_column_limit
		crew_column_limit = length(rc_routes)
	end


	## find variables we will be allowed to use

	# get keys in vector form
	plan_keys = [i for i in eachindex(rc_plans)]

	# get teduced costs in vector form
	plan_values = [rc_plans[ix] for ix in plan_keys]

	# get the indices of the first "fire_column_limit" indices of plan_values
	sorted_plan_ixs = Int.(plan_values .* 0)
	used_plan_ixs =
		[
			i for
			i in partialsortperm!(sorted_plan_ixs, plan_values, 1:fire_column_limit)
		]

	# get the max reduced cost selected
	max_plan_rc = plan_values[sorted_plan_ixs[fire_column_limit]]

	# if this reduced cost is too high for the upper bound
	if lp_objective + max_plan_rc > upper_bound + 1e-7

		# delete elements of used_plan_ixs
		to_delete = Int64[]
		for (i, ix) in enumerate(used_plan_ixs)
			if lp_objective + plan_values[ix] > upper_bound + 1e-7
				push!(to_delete, i)
			end
		end
		@debug "removing fire plans due to reduced-cost bound" length(to_delete)
		deleteat!(used_plan_ixs, to_delete)
	end

	# get the keys corresponding to these indices
	used_plan_keys = plan_keys[used_plan_ixs]
	unused_plan_keys = [i for i in plan_keys if i ∉ used_plan_keys]


	route_keys = [i for i in eachindex(rc_routes)]
	route_values = [rc_routes[ix] for ix in route_keys]
	sorted_route_ixs = Int.(route_values .* 0)
	used_route_ixs =
		[
			i for i in
			partialsortperm!(sorted_route_ixs, route_values, 1:crew_column_limit)
		]
	max_route_rc = route_values[sorted_route_ixs[crew_column_limit]]

	# if this reduced cost is too high for the upper bound
	if lp_objective + max_route_rc > upper_bound + 1e-7

		# delete elements of used_route_ixs
		to_delete = Int64[]
		for (i, ix) in enumerate(used_route_ixs)
			if lp_objective + route_values[ix] > upper_bound + 1e-7
				push!(to_delete, i)
			end
		end
		@debug "removing crew routes due to reduced-cost bound" length(to_delete)

		deleteat!(used_route_ixs, to_delete)
	end

	used_route_keys = route_keys[used_route_ixs]
	unused_route_keys = [i for i in route_keys if i ∉ used_route_keys]

	# fix unused variables to 0, set binary variables
	for i ∈ unused_route_keys
		if ~isnothing(warm_start_routes) && (i ∈ eachindex(warm_start_routes)) &&
		   (warm_start_routes[i] > 0.99)
			@warn "Route pruned by reduced cost bound but part of warm start solution, should not happen if warm start solution is best UB"
		end
		fix(solved_rmp.routes[i], 0, force = true)
	end
	for i ∈ unused_plan_keys
		if ~isnothing(warm_start_plans) && (i ∈ eachindex(warm_start_plans)) &&
		   (warm_start_plans[i] > 0.99)
			@warn "Plan pruned by reduced cost bound but part of warm start solution, should not happen if warm start solution is best UB"
		end
		fix(solved_rmp.plans[i], 0, force = true)
	end
	for i ∈ used_route_keys
		set_binary(solved_rmp.routes[i])
	end
	for i ∈ used_plan_keys
		set_binary(solved_rmp.plans[i])
	end

	if ~isnothing(warm_start_plans)
		@info "pushing solution to IP"
		plans_ws = 0
		for i ∈ eachindex(solved_rmp.plans)
			if (i ∈ eachindex(warm_start_plans)) && (warm_start_plans[i] > 0.99)
				plans_ws += 1
				set_start_value(solved_rmp.plans[i], 1)
			else
				set_start_value(solved_rmp.plans[i], 0)
			end
		end
		@debug "ws_plans" plans_ws
		routes_ws = 0
		for i ∈ eachindex(solved_rmp.routes)
			if (i ∈ eachindex(warm_start_routes)) && (warm_start_routes[i] > 0.99)
				routes_ws += 1
				set_start_value(solved_rmp.routes[i], 1)
			else
				set_start_value(solved_rmp.routes[i], 0)
			end
		end
		@debug "ws_routes" routes_ws

	end

	# set time TimeLimit and MIPFocus
	set_optimizer_attribute(solved_rmp.model, "TimeLimit", time_limit)
	set_optimizer_attribute(solved_rmp.model, "OutputFlag", 1)
	# solve
	optimize!(solved_rmp.model)

	# get values to return
	obj = Inf
	routes_used = nothing
	plans_used = nothing
	if result_count(solved_rmp.model) >= 1
		obj = objective_value(solved_rmp.model)
		routes_used = Int.(round.(value.(solved_rmp.routes)))
		plans_used = Int.(round.(value.(solved_rmp.plans)))
	end
	obj_bound = objective_bound(solved_rmp.model)


	return obj, obj_bound, routes_used, plans_used

end

function heuristic_upper_bound!!(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	explored_bb_node::BranchAndBoundNode,
	num_iters::Int,
	soft_time_limit::Float64,
	kill_if_no_improvement_rounds::Int64,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	cut_search_enumeration_limit::Int64,
	gurobi_env;
	price_and_cut_soft_time_limit,
	cut_loop_max,
	relative_improvement_cut_req,
	gub_cover_cuts,
	general_gub_cuts,
	single_fire_cuts,
	decrease_gub_allots,
	single_fire_lift,
	fires_to_ignore = Int64[],
	routes_best_sol = nothing,
	plans_best_sol = nothing,
)
	start_time = time()
	@info "Finding heuristic upper bound" explored_bb_node.ix

	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	cut_data = deepcopy(explored_bb_node.cut_data)

	# keep all columns from explored node
	crew_ixs =
		[
			[i[1] for i in eachindex(explored_bb_node.master_problem.routes[j, :])]
			for j ∈ 1:num_crews
		]
	fire_ixs =
		[
			[i[1] for i in eachindex(explored_bb_node.master_problem.plans[j, :])]
			for j ∈ 1:num_fires
		]
	if isempty(crew_ixs)
		crew_ixs = [Int64[] for crew in 1:num_crews]
	end
	if isempty(fire_ixs)
		fire_ixs = [Int64[] for fire in 1:num_fires]
	end

	# add in columns from best solution
	if ~isnothing(routes_best_sol)
		for (crew, route) ∈ eachindex(routes_best_sol)
			if value(routes_best_sol[(crew, route)]) > 0.99
				if route ∉ crew_ixs[crew]
					push!(crew_ixs[crew], route)
					@info "pushing route best sol"
				end
			end
		end

		for (fire, plan) ∈ eachindex(plans_best_sol)
			if value(plans_best_sol[(fire, plan)]) > 0.99
				if plan ∉ fire_ixs[fire]
					push!(fire_ixs[fire], plan)
					@info "pushing plan best sol"
				end
			end
		end
	end

	# grab any fire and crew branching rules
	crew_rules = CrewAssignmentBranchingRule[]
	fire_rules = FireDemandBranchingRule[]
	cur_node = explored_bb_node
	while ~isnothing(cur_node)

		crew_rules = vcat(cur_node.new_crew_branching_rules, crew_rules)
		fire_rules = vcat(cur_node.new_fire_branching_rules, fire_rules)
		cur_node = cur_node.parent
	end

	global_rules = GlobalFireAllotmentBranchingRule[]
	ub = Inf
	ub_rmp = nothing
	routes = routes_best_sol
	plans = plans_best_sol
	rounds_since_improvement = 0


	fractional_weighted_average_solution, _ =
		get_fire_and_crew_incumbent_weighted_average(explored_bb_node.master_problem,
			crew_routes,
			fire_plans,
		)

	current_allotment = Int.(ceil.(fractional_weighted_average_solution))


	for iter ∈ 1:num_iters

		cur_time = time() - start_time
		if cur_time > soft_time_limit
			@info "Breaking heuristic round" iter - 1 cur_time soft_time_limit
			break
		end


		# get rid of past cuts
		# do we want cuts? lose guarantee of feasibility in next step because we may find a cut later 
		# cut_data = CutData(num_crews, num_fires, num_time_periods)

		branching_rule =
			GlobalFireAllotmentBranchingRule(current_allotment,
				false,
				fire_plans,
				fire_subproblems,
			)
		global_rules = [branching_rule]

		for fire ∈ 1:num_fires
			to_delete = Int64[]
			for ix ∈ eachindex(fire_ixs[fire])
				plan_ix = fire_ixs[fire][ix]
				if (fire, plan_ix) ∈ keys(branching_rule.mp_lookup)
					push!(to_delete, ix)
				end
			end
			deleteat!(fire_ixs[fire], to_delete)
		end

		@info "entering heuristic round" branching_rule.allotment_matrix
		for rule in crew_rules
			@info "crew rule" rule crew_ixs
		end
		for rule in fire_rules
			@info "fire rule" rule fire_ixs
		end
		rmp = define_restricted_master_problem(
			gurobi_env,
			crew_routes,
			crew_ixs,
			fire_plans,
			fire_ixs,
			cut_data,
			global_rules,
			false,
			fires_to_ignore,
		)


		# TODO consider cut management
		t = @elapsed price_and_cut!!!!(
			rmp,
			crew_routes,
			fire_plans,
			cut_data,
			crew_subproblems,
			fire_subproblems,
			cut_search_enumeration_limit,
			crew_rules,
			fire_rules,
			global_rules,
			fires_to_ignore,
			soft_time_limit = price_and_cut_soft_time_limit,
			loop_max = cut_loop_max,
			relative_improvement_cut_req = relative_improvement_cut_req,
			gub_cover_cuts = gub_cover_cuts,
			general_gub_cuts = general_gub_cuts,
			single_fire_cuts = single_fire_cuts,
			decrease_gub_allots = decrease_gub_allots,
			single_fire_lift = single_fire_lift,
			upper_bound = ub,
			time_limit = 20.0)
		@info "Price and cut time (heuristic)" t

		# add in columns from best feasible solution so far
		if ~isnothing(routes)

			# this should usually be unnecessary since we don't often branch on crews
			for (crew, route) ∈ eachindex(routes)
				if (routes[(crew, route)] > 0.99) &&
				   ((crew, route) ∉ eachindex(rmp.routes))
					add_column_to_master_problem!!(
						rmp,
						cut_data,
						crew_routes,
						crew,
						route,
					)
					@warn "Adding crew column to heuristic from best feasible solution, why was it dropped?"
				end
			end

			for (fire, plan) ∈ eachindex(plans)
				if (plans[(fire, plan)] > 0.99) &&
				   ((fire, plan) ∉ eachindex(rmp.plans))
					add_column_to_master_problem!!(
						rmp,
						cut_data,
						fire_plans,
						global_rules,
						fire,
						plan,
					)
				end
			end
		end

		# remove capacity constraint to allow best feasible solution so far
		set_normalized_rhs.(rmp.fire_allotment_branches, -10000)
		optimize!(rmp.model)

		crew_ixs = [[i[1] for i in eachindex(rmp.routes[j, :])] for j ∈ 1:num_crews]
		fire_ixs = [[i[1] for i in eachindex(rmp.plans[j, :])] for j ∈ 1:num_fires]


		lb_node =
			rmp.termination_status == MOI.LOCALLY_SOLVED ?
			objective_value(rmp.model) : -Inf

		if lb_node / ub > 1 - 1e-9
			@debug "pruning heuristic round by bound, bumping allotments"
			current_allotment = current_allotment .+ 1
			continue
		end

		# TODO add global branching rule as a valid inequality to help solver find cuts

		t = @elapsed obj, obj_bound, routes, plans =
			find_integer_solution(
				rmp,
				ub,
				120000,
				400000,
				40.0,
				warm_start_plans = plans,
				warm_start_routes = routes,
			)

		rounds_since_improvement += 1
		if obj < ub - 1e-9
			ub = obj
			ub_rmp = rmp
			rounds_since_improvement = 0
		elseif obj == Inf && ub < Inf
			@warn "Failure of warm start solution"
		end
		@info "found sol" t obj obj_bound
		if rounds_since_improvement >= kill_if_no_improvement_rounds
			@info "Too long since improvement in heuristic, killing early" rounds_since_improvement
			break
		end

		fire_allots, _ =
			get_fire_and_crew_incumbent_weighted_average(ub_rmp,
				crew_routes,
				fire_plans,
			)

		@debug "solution bounds" t lb_node ub obj obj_bound fire_allots

		current_allotment = current_allotment .+ 1
	end
	@info "Found heuristic upper bound" ub

	return ub, ub_rmp, routes, plans
end


function explore_node!!(
	branch_and_bound_node::BranchAndBoundNode,
	all_nodes::Vector{BranchAndBoundNode},
	current_global_upper_bound::Float64,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	cut_search_enumeration_limit::Int64,
	warm_start_strategy::Union{String, Nothing},
	gurobi_env;
	branching_strategy,
	price_and_cut_soft_time_limit,
	cut_loop_max,
	relative_improvement_cut_req,
	gub_cover_cuts,
	general_gub_cuts,
	single_fire_cuts,
	decrease_gub_allots,
	single_fire_lift,
	log_cuts_file,
	fires_to_ignore,
	rel_tol = 1e-9)

	@info "Exploring node" branch_and_bound_node.ix fires_to_ignore

	deferral_stabilization = false
	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)
	cut_data = branch_and_bound_node.cut_data
	## get the columns with which to initialize restricted master problem

	# if we are at the root node, there are no columns yet, and stabilization applies
	if isnothing(branch_and_bound_node.parent)

		# TODO in sequential optimization, we can use the solution found at the prior time step
		crew_ixs = [Int[] for i ∈ 1:num_crews]
		fire_ixs = [Int[] for i ∈ 1:num_fires]
		deferral_stabilization = true

	else
		# if we are not at the root node, there are a lot of options here, but
		# for now take all the columns generated by the parent RMP that satisfy 
		# the new branching rule. Maybe could improve performance by culling
		# columns based on reduced costs or absence in basis, but first try at this
		# showed worse performance

		cull = false
		parent_rmp = branch_and_bound_node.parent.master_problem
		if ~cull
			# TODO this should be cleaner with new ix data structure kept with rmp
			crew_ixs =
				[
					[i[1] for i in eachindex(parent_rmp.routes[j, :])] for
					j ∈ 1:num_crews
				]
			fire_ixs =
				[
					[i[1] for i in eachindex(parent_rmp.plans[j, :])] for
					j ∈ 1:num_fires
				]
		else
			# TODO if we want to cull, need some better solution of timing of restore_integrality
			# in parent node
			@warn "If we restored integrality in parent node, only getting these columns"
			crew_ixs =
				[
					[
						i[1] for i in eachindex(parent_rmp.routes[j, :]) if
						value(parent_rmp.routes[j, i...]) > 1e-4
					] for j ∈ 1:num_crews
				]
			fire_ixs =
				[
					[
						i[1] for i in eachindex(parent_rmp.plans[j, :]) if
						value(parent_rmp.plans[j, i...]) > 1e-4
					] for j ∈ 1:num_fires
				]
		end
		@debug "num ix" length(fire_ixs[1])

		for rule in branch_and_bound_node.new_crew_branching_rules
			crew_ixs = apply_branching_rule(crew_ixs, crew_routes, rule)
		end
		for rule in branch_and_bound_node.new_fire_branching_rules
			fire_ixs = apply_branching_rule(fire_ixs, fire_plans, rule)
		end

		# remove plans excluded by left branching rules
		for rule in branch_and_bound_node.new_global_fire_allotment_branching_rules
			if ~rule.geq_flag
				for fire in 1:num_fires
					fire_ixs[fire] = filter!(
						x -> (fire, x) ∉ keys(rule.mp_lookup),
						fire_ixs[fire],
					)
				end
			end
		end
	end


	# get the branching rules
	crew_rules = CrewAssignmentBranchingRule[]
	fire_rules = FireDemandBranchingRule[]
	global_rules = GlobalFireAllotmentBranchingRule[]

	cur_node = branch_and_bound_node
	while ~isnothing(cur_node)

		crew_rules = vcat(cur_node.new_crew_branching_rules, crew_rules)
		fire_rules = vcat(cur_node.new_fire_branching_rules, fire_rules)
		global_rules =
			vcat(cur_node.new_global_fire_allotment_branching_rules, global_rules)
		cur_node = cur_node.parent

	end
	@info "rules" crew_rules fire_rules global_rules

	# define the restricted master problem
	## TODO how do we handle existing cuts
	## currently carrying them all

	t = @elapsed rmp = define_restricted_master_problem(
		gurobi_env,
		crew_routes,
		crew_ixs,
		fire_plans,
		fire_ixs,
		cut_data,
		global_rules,
		deferral_stabilization,
		fires_to_ignore,
	)
	@info "Define rmp time (b-and-b)" t

	t = @elapsed price_and_cut!!!!(
		rmp,
		crew_routes,
		fire_plans,
		cut_data,
		crew_subproblems,
		fire_subproblems,
		cut_search_enumeration_limit,
		crew_rules,
		fire_rules,
		global_rules,
		fires_to_ignore,
		soft_time_limit = price_and_cut_soft_time_limit,
		loop_max = cut_loop_max,
		relative_improvement_cut_req = relative_improvement_cut_req,
		upper_bound = current_global_upper_bound,
		gub_cover_cuts = gub_cover_cuts,
		general_gub_cuts = general_gub_cuts,
		single_fire_cuts = single_fire_cuts,
		decrease_gub_allots = decrease_gub_allots,
		single_fire_lift = single_fire_lift,
		log_progress_file = log_cuts_file)

	@info "Price and cut time (b-and-b)" t
	@debug "after price and cut" objective_value(rmp.model) crew_routes.routes_per_crew fire_plans.plans_per_fire

	# extract some data
	binding_cuts = [
		i for i in eachindex(rmp.gub_cover_cuts) if
		dual(rmp.gub_cover_cuts[i]) > 1e-4
	]
	@debug "cut_duals" dual.(rmp.gub_cover_cuts)
	used_plans = [
		i for i in eachindex(rmp.plans) if
		value(rmp.plans[i]) > 1e-4
	]
	used_routes = [
		i for i in eachindex(rmp.routes) if
		value(rmp.routes[i]) > 1e-4
	]

	# update the rmp 
	branch_and_bound_node.master_problem = rmp

	# all_fire_allots, all_crew_allots = extract_usages(crew_routes, fire_plans, rmp)
	# @debug "usages" all_fire_allots all_crew_allots

	# update the branch-and-bound node to be feasible or not
	if rmp.termination_status == MOI.INFEASIBLE
		branch_and_bound_node.feasible = false
		branch_and_bound_node.l_bound = Inf
		@debug "infeasible node" node_ix
	else
		if rmp.termination_status ∉ [MOI.LOCALLY_SOLVED, MOI.OBJECTIVE_LIMIT]
			@debug "Node not feasible or optimal, CG did not terminate?"
			error("Node not feasible or optimal, CG did not terminate?")
		else
			branch_and_bound_node.l_bound = objective_value(rmp.model)
		end
		branch_and_bound_node.feasible = true

		plan_values = value.(rmp.plans)
		route_values = value.(rmp.routes)

		tolerance = 1e-8
		integer =
			all((plan_values .< tolerance) .|| (plan_values .> 1 - tolerance)) &&
			all((route_values .< tolerance) .|| (route_values .> 1 - tolerance))
		if integer
			branch_and_bound_node.u_bound = branch_and_bound_node.l_bound
		end
	end

	# if we cannot prune
	if (branch_and_bound_node.u_bound - branch_and_bound_node.l_bound > rel_tol) &
	   branch_and_bound_node.feasible &
	   (branch_and_bound_node.l_bound / current_global_upper_bound < 1 - rel_tol)

		# TODO think about branching rules
		branch_type, branch_ix, var_variance, var_mean =
			nothing, nothing, nothing, nothing
		if branching_strategy == "linking_dual_max_variance"
			# decide the next branching rules
			branch_type, branch_ix, var_variance, var_mean =
				max_variance_natural_variable(
					crew_routes,
					fire_plans,
					route_values,
					plan_values,
					dual.(rmp.supply_demand_linking) .+ 1e-5,
				)
		elseif branching_strategy == "max_variance"
			branch_type, branch_ix, var_variance, var_mean =
				max_variance_natural_variable(
					crew_routes,
					fire_plans,
					route_values,
					plan_values,
					nothing,
				)
		elseif branching_strategy == "most_fractional"
			branch_type, branch_ix, var_variance, var_mean =
				most_fractional_natural_variable(
					crew_routes,
					fire_plans,
					route_values,
					plan_values,
					nothing,
				)
		else
			error("branching strategy not implemented")
		end

		# restrict to used cuts
		used_cuts = restrict_CutData(cut_data, binding_cuts)
		@debug "cuts" used_cuts.cut_dict

		@assert var_variance > 0 "Cannot branch on variable with no variance, should already be integral"

		# create two new nodes with branching rules
		if branch_type == "fire"
			left_branching_rule = FireDemandBranchingRule(
				Tuple(branch_ix)...,
				Int(floor(var_mean)),
				"less_than_or_equal",
				Int64[],
			)
			right_branching_rule = FireDemandBranchingRule(
				Tuple(branch_ix)...,
				Int(floor(var_mean)) + 1,
				"greater_than_or_equal",
				Int64[],
			)

			get_branching_rule_fire_prohibited_arcs!(left_branching_rule, fire_subproblems[Tuple(branch_ix)[1]].long_arcs)
			get_branching_rule_fire_prohibited_arcs!(right_branching_rule, fire_subproblems[Tuple(branch_ix)[1]].long_arcs)
			
			left_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 1,
				parent = branch_and_bound_node,
				new_fire_branching_rules = [left_branching_rule],
				cut_data = used_cuts,
			)
			right_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 2,
				parent = branch_and_bound_node,
				new_fire_branching_rules = [right_branching_rule],
				cut_data = used_cuts,
			)
			push!(all_nodes, left_child)
			push!(all_nodes, right_child)
			branch_and_bound_node.children = [left_child, right_child]


		else
			left_branching_rule = CrewAssignmentBranchingRule(
				Tuple(branch_ix)...,
				false,
				Int64[],
			)
			right_branching_rule = CrewAssignmentBranchingRule(
				Tuple(branch_ix)...,
				true,
				Int64[],
			)

			get_branching_rule_crew_prohibited_arcs!(left_branching_rule, crew_subproblems[Tuple(branch_ix)[1]].long_arcs)
			get_branching_rule_crew_prohibited_arcs!(right_branching_rule, crew_subproblems[Tuple(branch_ix)[1]].long_arcs)
			
			left_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 1,
				parent = branch_and_bound_node,
				new_crew_branching_rules = [left_branching_rule],
				cut_data = used_cuts,
			)
			right_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 2,
				parent = branch_and_bound_node,
				new_crew_branching_rules = [right_branching_rule],
				cut_data = used_cuts,
			)
			push!(all_nodes, left_child)
			push!(all_nodes, right_child)
			branch_and_bound_node.children = [left_child, right_child]

		end
		@info "branching rules" left_branching_rule right_branching_rule
	end

	return used_plans, used_routes, binding_cuts
end
