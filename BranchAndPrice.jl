include("CommonStructs.jl")

using Gurobi, Statistics
# const GRB_ENV = Gurobi.Env()



function max_variance_natural_variable(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	route_values::JuMP.Containers.SparseAxisArray,
	plan_values::JuMP.Containers.SparseAxisArray,
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
	@debug "Means" crew_means # crew_means
	@debug "Variances" crew_variances # crew_variances
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
		if plan_values[ix] > 0.0001
			@debug "used plan" ix plan_values[ix] fire_plans.crews_present[fire, plan, :]
		end
	end
	fire_variances = fire_sq_means - (fire_means .^ 2)
	@debug "Means" fire_means # crew_means
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
		old_avail_ixs = crew_avail_ixs[crew]
		satisfy_ixs = [
			i for i in old_avail_ixs if satisfies_branching_rule(
				branching_rule,
				crew_routes.fires_fought[branching_rule.crew_ix, i, :, :],
			)
		]
		push!(output, satisfy_ixs)
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
		old_avail_ixs = fire_avail_ixs[fire]
		satisfy_ixs = [
			i for i in old_avail_ixs if satisfies_branching_rule(
				branching_rule,
				fire_plans.crews_present[branching_rule.fire_ix, i, :],
			)
		]
		push!(output, satisfy_ixs)
	end


	return output

end
function explore_node!!(
	branch_and_bound_node::BranchAndBoundNode,
	all_nodes::Vector{BranchAndBoundNode},
	current_global_upper_bound::Float64,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	warm_start_strategy::Union{String, Nothing},
	gurobi_env)

	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	## get the columns with which to initialize restricted master problem

	# if we are at the root node, there are no columns yet
	if isnothing(branch_and_bound_node.parent)
		crew_ixs = [Int[] for i ∈ 1:num_crews]
		fire_ixs = [Int[] for i ∈ 1:num_fires]

		# if we are not at the root node, there are a lot of options here, but
		# for now take all the columns generated by the parent RMP that satisfy 
		# the new branching rule. Probably could improve performance by culling
		# columns based on reduced costs or absence in basis.
	else
		parent_rmp = branch_and_bound_node.parent.master_problem
		crew_ixs =
			[[i[1] for i in eachindex(parent_rmp.routes[j, :])] for j ∈ 1:num_crews]
		fire_ixs =
			[[i[1] for i in eachindex(parent_rmp.plans[j, :])] for j ∈ 1:num_fires]
		for rule in branch_and_bound_node.new_crew_branching_rules
			crew_ixs = apply_branching_rule(crew_ixs, crew_routes, rule)
		end
		for rule in branch_and_bound_node.new_fire_branching_rules
			fire_ixs = apply_branching_rule(fire_ixs, fire_plans, rule)
		end
	end

	# define the restricted master problem
	rmp = define_restricted_master_problem(
		gurobi_env,
		crew_routes,
		crew_ixs,
		fire_plans,
		fire_ixs,
	)

	# get the branching rules
	crew_rules = CrewSupplyBranchingRule[]
	fire_rules = FireDemandBranchingRule[]

	cur_node = branch_and_bound_node
	while ~isnothing(cur_node)

		crew_rules = vcat(cur_node.new_crew_branching_rules, crew_rules)
		fire_rules = vcat(cur_node.new_fire_branching_rules, fire_rules)
		cur_node = cur_node.parent

	end
	@debug "all branching rules found to pass to DCG" crew_rules fire_rules

	# run DCG, adding columns as needed
	double_column_generation!(
		rmp,
		crew_subproblems,
		fire_subproblems,
		crew_rules,
		fire_rules,
		crew_routes,
		fire_plans,
	)

	# update the rmp fire_flow_duals
	branch_and_bound_node.master_problem = rmp

	# update the branch-and-bound node to be feasible or not
	if rmp.termination_status == MOI.INFEASIBLE
		branch_and_bound_node.feasible = false
		branch_and_bound_node.integer = false
		branch_and_bound_node.l_bound = Inf
		println("infeasible here")
	else
		branch_and_bound_node.feasible = true
		branch_and_bound_node.l_bound = objective_value(rmp.model)

		# update the branch-and-bound node to be integer or not
		tolerance = 1e-4
		plan_values = value.(rmp.plans)
		route_values = value.(rmp.routes)
		integer =
			all((plan_values .< tolerance) .| (plan_values .> 1 - tolerance)) &
			all((route_values .< tolerance) .| (route_values .> 1 - tolerance))
		branch_and_bound_node.integer = integer
	end

	# if we cannot prune
	if ~branch_and_bound_node.integer & branch_and_bound_node.feasible &
	   (branch_and_bound_node.l_bound < current_global_upper_bound)

		# decide the next branching rules
		branch_type, branch_ix, var_variance, var_mean =
			max_variance_natural_variable(
				crew_routes,
				fire_plans,
				route_values,
				plan_values,
			)

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
			)
			right_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 2,
				parent = branch_and_bound_node,
				new_fire_branching_rules = [right_branching_rule],
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
			)
			right_child = BranchAndBoundNode(
				ix = size(all_nodes)[1] + 2,
				parent = branch_and_bound_node,
				new_crew_branching_rules = [right_branching_rule],
			)
			push!(all_nodes, left_child)
			push!(all_nodes, right_child)
			branch_and_bound_node.children = [left_child, right_child]

		end
		@debug "branching rules" left_branching_rule right_branching_rule
	end
end




function test_BranchAndBoundNode()

	bb_node = BranchAndBoundNode(ix = 1, parent = nothing)
	println(bb_node)

end

# test_BranchAndBoundNode()
