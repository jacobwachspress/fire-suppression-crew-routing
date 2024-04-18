include("CommonStructs.jl")
include("BranchingRules.jl")
include("GUBKnapsackCoverCuts.jl")

using Gurobi, Statistics, JSON

mutable struct BranchAndBoundNode

	const ix::Int64
	const parent::Union{Nothing, BranchAndBoundNode}
	const new_crew_branching_rules::Vector{CrewSupplyBranchingRule}
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
	new_crew_branching_rules::Vector{CrewSupplyBranchingRule} = CrewSupplyBranchingRule[],
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

function branch_and_price(
    num_fires::Int,
    num_crews::Int,
    num_time_periods::Int;
    max_nodes=10000,
    algo_tracking=false,
	branching_strategy="linking_dual_max_variance",
    gub_cut_limit_per_time=100000,
    cut_loop_max=25,
	price_and_cut_soft_time_limit=180.0,
    relative_improvement_cut_req=1e-10,
    soft_heuristic_time_limit=300.0,
    hard_heuristic_iteration_limit=10,
    heuristic_must_improve_rounds=2,
    heuristic_cadence=5,
    total_time_limit=1800.0,
    bb_node_gub_cover_cuts=true,
    bb_node_general_gub_cuts=true,
    bb_node_single_fire_cuts=false,
    bb_node_decrease_gub_allots=true,
    bb_node_single_fire_lift=true,
    heuristic_gub_cover_cuts=true,
    heuristic_general_gub_cuts=true,
    heuristic_single_fire_cuts=false,
    heuristic_decrease_gub_allots=true,
    heuristic_single_fire_lift=true,
    price_and_cut_file=nothing
)
    start_time = time()

    @info "Initializing data structures"
    # initialize input data
    crew_routes, fire_plans, crew_models, fire_models, cut_data =
        initialize_data_structures(num_fires, num_crews, num_time_periods)


    algo_tracking ?
    (@info "Checkpoint after initializing data structures" time() - start_time) :
    nothing

    explored_nodes = []
    ubs = []
    lbs = []
    columns = []
    times = []
    time_1 = time() - start_time

    # initialize nodes list with the root node
    nodes = BranchAndBoundNode[]
    first_node = BranchAndBoundNode(ix=1, parent=nothing, cut_data=cut_data)
    push!(nodes, first_node)

    # initialize global variables to track in branch-and-bound tree
    ub = Inf
	lb = 0
    ub_ix::Int = -1
    routes_best_sol = nothing
    plans_best_sol = nothing

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
            gub_cut_limit_per_time,
            nothing,
            GRB_ENV,
			price_and_cut_soft_time_limit=price_and_cut_soft_time_limit,
            cut_loop_max=cut_loop_max,
            relative_improvement_cut_req=relative_improvement_cut_req,
            gub_cover_cuts=bb_node_gub_cover_cuts,
            general_gub_cuts=bb_node_general_gub_cuts,
            single_fire_cuts=bb_node_single_fire_cuts,
            decrease_gub_allots=bb_node_decrease_gub_allots,
            single_fire_lift=bb_node_single_fire_lift,
            log_cuts_file=price_and_cut_file,
			branching_strategy=branching_strategy
        )

        if time() - start_time > total_time_limit
            @info "Full time limit reached"
            break
        end

        if (node_explored_count % heuristic_cadence == 1) &&
           (soft_heuristic_time_limit > 0.0) && (lb / ub < 1 - 1e-3) # gurobi has 1e-4 tol I can't fix
            heuristic_ub, ub_rmp, routes_best_sol, plans_best_sol = heuristic_upper_bound!!(
                crew_routes,
                fire_plans,
                nodes[node_ix],
                hard_heuristic_iteration_limit,
                soft_heuristic_time_limit,
                heuristic_must_improve_rounds,
                crew_models,
                fire_models,
                gub_cut_limit_per_time,
                GRB_ENV,
                routes_best_sol=routes_best_sol,
                plans_best_sol=plans_best_sol,
				price_and_cut_soft_time_limit=price_and_cut_soft_time_limit,
                cut_loop_max=cut_loop_max,
                relative_improvement_cut_req=relative_improvement_cut_req,
                gub_cover_cuts=heuristic_gub_cover_cuts,
                general_gub_cuts=heuristic_general_gub_cuts,
                single_fire_cuts=heuristic_single_fire_cuts,
                decrease_gub_allots=heuristic_decrease_gub_allots,
                single_fire_lift=heuristic_single_fire_lift
            )

            if time() - start_time > total_time_limit
                @info "Full time limit reached"
                break
            end

            nodes[node_ix].heuristic_found_master_problem = ub_rmp

            if heuristic_ub < nodes[node_ix].u_bound
                nodes[node_ix].u_bound = heuristic_ub
            end
        end

        ## now that we have done the heuristic, can mess up rmp and lose duals

		# if (ub / nodes[node_ix].l_bound < 1.001) 
		# 	t = @elapsed obj, obj_bound, routes, plans =
		# 		find_integer_solution(
		# 			nodes[node_ix].master_problem,
		# 			ub,
		# 			12000,
		# 			40000,
		# 			10.0,)			
		# 	@info "restore_integrality" t obj obj_bound
		# 	if obj < nodes[node_ix].u_bound
		# 		nodes[node_ix].u_bound = obj
		# 	end
		# end


        # if this node has an integer solution, check if we have found 
        # a better solution than the incumbent
        # TODO keep track if it comes from heurisitc or no
        if nodes[node_ix].u_bound < ub
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

    return explored_nodes, ubs, lbs, columns, times, time_1

end

function initialize_data_structures(
    num_fires::Int64,
    num_crews::Int64,
    num_time_periods::Int64,
)
    crew_models = build_crew_models(
        "data/raw/big_fire",
        num_fires,
        num_crews,
        num_time_periods,
    )

    fire_models = build_fire_models(
        "data/raw/big_fire",
        num_fires,
        num_crews,
        num_time_periods,
    )


    crew_routes = CrewRouteData(100000, num_fires, num_crews, num_time_periods)
    fire_plans = FirePlanData(100000, num_fires, num_time_periods)
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

function max_variance_natural_variable(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	route_values::JuMP.Containers.SparseAxisArray,
	plan_values::JuMP.Containers.SparseAxisArray,
	linking_duals::Union{Nothing, Matrix{Float64}}
)
	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	# calculate variance of A_{cgt}, crew c suppressing fire g at time t
	crew_means = zeros(Float64, (num_crews, num_fires, num_time_periods))
	for ix ∈ eachindex(route_values)
		crew = ix[1]
		route = ix[2]
		crew_means[crew, :, :] +=
			route_values[ix] * crew_routes.fires_fought[crew, route, :, :]
	end
	crew_variances = crew_means .* (1 .- crew_means)

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
		for j ∈ 1:num_crews
			for g ∈ 1:num_fires
				for t ∈ 1:num_time_periods
					crew_variances[j, g, t] *= ((linking_duals[g, t]) ^ 2)
				end
			end
		end

		for g ∈ 1:num_fires
			for t ∈ 1:num_time_periods
				fire_variances[g, t] *= ((linking_duals[g, t]) ^ 2)
			end
		end
	end

	@debug "Variances" fire_variances # crew_variances
	# get the max variance for each natural variable type
	crew_max_var, crew_max_ix = findmax(crew_variances)
	fire_max_var, fire_max_ix = findmax(fire_variances)

	# return the info needed to create the branching rule
	if fire_max_var > crew_max_var
		return "fire", fire_max_ix, fire_max_var, fire_means[fire_max_ix]
	else
		return "crew", crew_max_ix, crew_max_var, crew_means[crew_max_ix]
	end

end


function apply_branching_rule(
	crew_avail_ixs::Vector{Vector{Int64}},
	crew_routes::CrewRouteData,
	branching_rule::CrewSupplyBranchingRule)

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

function price_and_cut!!(
	rmp::RestrictedMasterProblem,
	cut_data::CutData,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	gub_cut_limit_per_time::Int64,
	crew_rules::Vector{CrewSupplyBranchingRule},
	fire_rules::Vector{FireDemandBranchingRule},
	global_fire_allotment_rules::Vector{GlobalFireAllotmentBranchingRule},
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData;
	gub_cover_cuts,
	general_gub_cuts,
	single_fire_cuts,
	decrease_gub_allots,
	single_fire_lift,
	soft_time_limit,
	loop_max,
	relative_improvement_cut_req,
	upper_bound::Float64 = 1e20,
	log_progress_file = nothing,
	)

	if ~isnothing(log_progress_file)
		ts = []
		objs = []
		num_routes = []
		num_plans = []
		num_cuts_found = []
		active_cuts = []
	end

	t = time()
	loop_ix = 1
	most_recent_obj = 0

	while loop_ix < loop_max

		# run DCG, adding columns as needed
		double_column_generation!(
			rmp,
			crew_subproblems,
			fire_subproblems,
			crew_rules,
			fire_rules,
			global_fire_allotment_rules,
			crew_routes,
			fire_plans,
			cut_data,
			upper_bound = upper_bound,
		)
		if rmp.termination_status == MOI.OBJECTIVE_LIMIT
			@debug "no more cuts needed"
			break
		end

		current_obj = objective_value(rmp.model)
		if ~isnothing(log_progress_file)
			push!(ts, time() - t)
			push!(objs, current_obj)
			push!(num_routes, sum(crew_routes.routes_per_crew))
			push!(num_plans, sum(fire_plans.plans_per_fire))
			push!(num_cuts_found, sum(cut_data.cuts_per_time))
			push!(active_cuts, sum([1 for i in eachindex(rmp.gub_cover_cuts) if dual(rmp.gub_cover_cuts[i]) > 1e-4]))
		end

		@debug "in cut generation" loop_ix current_obj
		if most_recent_obj / current_obj > 1 - relative_improvement_cut_req
			@info "halting cut generation early, too small improvement" loop_ix
			break
		end
		most_recent_obj = current_obj

		if time() - t > soft_time_limit
			@info "cut time limit" soft_time_limit
			break
		end

		# add cuts
		num_cuts = find_and_incorporate_knapsack_gub_cuts!!(
			cut_data,
			gub_cut_limit_per_time,
			rmp,
			crew_routes,
			fire_plans,
			crew_subproblems,
			fire_subproblems,
			gub_cover_cuts = gub_cover_cuts,
			general_gub_cuts = general_gub_cuts,
			single_fire_cuts = single_fire_cuts,
			decrease_gub_allots = decrease_gub_allots,
			single_fire_lift = single_fire_lift
		)
		@debug "after cuts" num_cuts objective_value(rmp.model)
		if (num_cuts == 0)
			@debug "No cuts found, breaking" loop_ix
			break
		end

		if num_cuts == 0
			@debug "No cuts found, breaking" loop_ix
			break

		end

		loop_ix += 1

	end

	# if we got completely through all the loop of cut generation
	# TODO can skip if time limit hit before cuts, should not really matter
	if loop_ix == loop_max
		@info "There may be more cuts; one last DCG to have provable lower bound" loop_ix
		double_column_generation!(
			rmp,
			crew_subproblems,
			fire_subproblems,
			crew_rules,
			fire_rules,
			global_fire_allotment_rules,
			crew_routes,
			fire_plans,
			cut_data,
			upper_bound = upper_bound,
		)

		current_obj = objective_value(rmp.model)
		if ~isnothing(log_progress_file)
			push!(ts, time() - t)
			push!(objs, current_obj)
			push!(num_routes, sum(crew_routes.routes_per_crew))
			push!(num_plans, sum(fire_plans.plans_per_fire))
			push!(num_cuts_found, sum(cut_data.cuts_per_time))
			push!(active_cuts, sum([1 for i in eachindex(rmp.gub_cover_cuts) if dual(rmp.gub_cover_cuts[i]) > 1e-4]))

		end
	end

	if ~isnothing(log_progress_file)
		outputs = Dict(
			"times" => ts,
			"objectives" => objs,
			"num_routes" => num_routes,
			"num_plans" => num_plans,
			"num_cuts" => num_cuts_found,
			"num_active_cuts" => active_cuts
		)
		open(log_progress_file, "w") do f
			JSON.print(f, outputs, 4)
		end
	end

	@info "After price-and-cut" dual.(rmp.supply_demand_linking)

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
	if lp_objective + max_plan_rc > upper_bound + 1e-5

		# delete elements of used_plan_ixs
		to_delete = Int64[]
		for (i, ix) in enumerate(used_plan_ixs)
			if lp_objective + plan_values[ix] > upper_bound + 1e-5
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
	if lp_objective + max_route_rc > upper_bound + 1e-5

		# delete elements of used_route_ixs
		to_delete = Int64[]
		for (i, ix) in enumerate(used_route_ixs)
			if lp_objective + route_values[ix] > upper_bound + 1e-5
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
		if ~isnothing(warm_start_plans) && (i ∈ eachindex(warm_start_plans)) && (warm_start_plans[i] > 0.99)
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
	gub_cut_limit_per_time::Int64,
	gurobi_env;
	price_and_cut_soft_time_limit,
	cut_loop_max,
	relative_improvement_cut_req,
	gub_cover_cuts,
	general_gub_cuts,
	single_fire_cuts,
	decrease_gub_allots,
	single_fire_lift,
	routes_best_sol = nothing,
	plans_best_sol = nothing
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
	crew_rules = CrewSupplyBranchingRule[]
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
			@info "Breaking heuristic round" iter-1 cur_time soft_time_limit
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

		@debug "entering heuristic round" branching_rule.allotment_matrix
		rmp = define_restricted_master_problem(
			gurobi_env,
			crew_routes,
			crew_ixs,
			fire_plans,
			fire_ixs,
			cut_data,
			global_rules,
		)


		# TODO consider cut management
		t = @elapsed price_and_cut!!(
			rmp,
			cut_data,
			crew_subproblems,
			fire_subproblems,
			gub_cut_limit_per_time,
			crew_rules,
			fire_rules,
			global_rules,
			crew_routes,
			fire_plans,
			soft_time_limit=price_and_cut_soft_time_limit,
			loop_max=cut_loop_max,
			relative_improvement_cut_req=relative_improvement_cut_req,
			gub_cover_cuts=gub_cover_cuts,
			general_gub_cuts=general_gub_cuts,
			single_fire_cuts=single_fire_cuts,
			decrease_gub_allots=decrease_gub_allots,
			single_fire_lift=single_fire_lift,
			upper_bound = ub)
		@info "Price and cut time (heuristic)" t 

		# add in columns from best feasible solution so far
		if ~isnothing(routes)

			# this should usually be unnecessary since we don't often branch on crews
			for (crew, route) ∈ eachindex(routes)
				if (routes[(crew, route)] > 0.99) && ((crew, route) ∉ eachindex(rmp.routes))
					add_column_to_master_problem!!(rmp, cut_data, crew_routes, crew, route)
					@warn "Adding crew column to heuristic from best feasible solution, why was it dropped?"
				end
			end

			for (fire, plan) ∈ eachindex(plans)
				if (plans[(fire, plan)] > 0.99) && ((fire, plan) ∉ eachindex(rmp.plans))
					add_column_to_master_problem!!(rmp, cut_data, fire_plans, global_rules, fire, plan) 
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
	gub_cut_limit_per_time::Int64,
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
	rel_tol = 1e-9)

	@info "Exploring node" branch_and_bound_node.ix

	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)
	cut_data = branch_and_bound_node.cut_data
	## get the columns with which to initialize restricted master problem

	# if we are at the root node, there are no columns yet
	if isnothing(branch_and_bound_node.parent)
		crew_ixs = [Int[] for i ∈ 1:num_crews]
		fire_ixs = [Int[] for i ∈ 1:num_fires]


	else
		# if we are not at the root node, there are a lot of options here, but
		# for now take all the columns generated by the parent RMP that satisfy 
		# the new branching rule. Maybe could improve performance by culling
		# columns based on reduced costs or absence in basis, but first try at this
		# showed worse performance

		cull = false
		parent_rmp = branch_and_bound_node.parent.master_problem
		if ~cull
			crew_ixs =
				[[i[1] for i in eachindex(parent_rmp.routes[j, :])] for j ∈ 1:num_crews]
			fire_ixs =
				[[i[1] for i in eachindex(parent_rmp.plans[j, :])] for j ∈ 1:num_fires]
		else
			# TODO if we want to cull, need some better solution of timing of restore_integrality
			# in parent node
			@warn "If we restored integrality in parent node, only getting these columns"
			crew_ixs =
			[[i[1] for i in eachindex(parent_rmp.routes[j, :]) if value(parent_rmp.routes[j, i...]) > 1e-4] for j ∈ 1:num_crews]
			fire_ixs =
			[[i[1] for i in eachindex(parent_rmp.plans[j, :]) if value(parent_rmp.plans[j, i...])  > 1e-4] for j ∈ 1:num_fires]
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
	crew_rules = CrewSupplyBranchingRule[]
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
	@debug "rules" crew_rules fire_rules global_rules

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
	)
	@info "Define rmp time (b-and-b)" t

	t = @elapsed price_and_cut!!(
		rmp,
		cut_data,
		crew_subproblems,
		fire_subproblems,
		gub_cut_limit_per_time,
		crew_rules,
		fire_rules,
		global_rules,
		crew_routes,
		fire_plans,
		soft_time_limit=price_and_cut_soft_time_limit,
		loop_max=cut_loop_max,
		relative_improvement_cut_req=relative_improvement_cut_req,
		upper_bound = current_global_upper_bound,
		gub_cover_cuts=gub_cover_cuts,
		general_gub_cuts=general_gub_cuts,
		single_fire_cuts=single_fire_cuts,
		decrease_gub_allots=decrease_gub_allots,
		single_fire_lift=single_fire_lift,
		log_progress_file=log_cuts_file)

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

	all_fire_allots, all_crew_allots = extract_usages(crew_routes, fire_plans, rmp)
	@info "usages" all_fire_allots all_crew_allots

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
		branch_type, branch_ix, var_variance, var_mean = nothing, nothing, nothing, nothing
		if branching_strategy == "linking_dual_max_variance"
			@info "duals" dual.(rmp.supply_demand_linking)
			# decide the next branching rules
			branch_type, branch_ix, var_variance, var_mean =
				max_variance_natural_variable(
					crew_routes,
					fire_plans,
					route_values,
					plan_values,
					dual.(rmp.supply_demand_linking) .+ 1e-5
				)
		elseif branching_strategy == "max_variance"
			branch_type, branch_ix, var_variance, var_mean =
			max_variance_natural_variable(
				crew_routes,
				fire_plans,
				route_values,
				plan_values,
				nothing
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
			)
			right_branching_rule = FireDemandBranchingRule(
				Tuple(branch_ix)...,
				Int(floor(var_mean)) + 1,
				"greater_than_or_equal",
			)
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
			left_branching_rule = CrewSupplyBranchingRule(
				Tuple(branch_ix)...,
				false,
			)
			right_branching_rule = CrewSupplyBranchingRule(
				Tuple(branch_ix)...,
				true,
			)
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