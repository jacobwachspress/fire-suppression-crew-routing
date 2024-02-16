include("CommonStructs.jl")
include("Subproblems.jl")

using JuMP

@kwdef struct TimeSpaceNetwork # always make time the first index

	arc_costs::Vector{Float64}
	state_in_arcs::Matrix{Vector{Int64}}

	# not quite sure how much this helps performance, but storing 2 copies
	# of array data allows always column-wise access
	crew_arc_array_long::Union{Nothing, Matrix{Int64}} = nothing
	crew_arc_array_wide::Union{Nothing, Matrix{Int64}} = nothing
	fire_arc_array_long::Union{Nothing, Matrix{Int64}} = nothing
	fire_arc_array_wide::Union{Nothing, Matrix{Int64}} = nothing

end

mutable struct RestrictedMasterProblem # TODO can make some JuMP things const?

	# model
	model::JuMP.Model

	# vars
	routes::JuMP.Containers.SparseAxisArray{VariableRef, 2, Tuple{Int64, Int64}}
	plans::JuMP.Containers.SparseAxisArray{VariableRef, 2, Tuple{Int64, Int64}}

	# constraints
	route_per_crew::Vector{ConstraintRef{Model, MOI.ConstraintIndex}}
	plan_per_fire::Vector{ConstraintRef{Model, MOI.ConstraintIndex}}
	supply_demand_linking::Matrix{ConstraintRef{Model, MOI.ConstraintIndex}}
	linking_perturbation::Matrix{ConstraintRef{Model, MOI.ConstraintIndex}}

	# termination status
	termination_status::MOI.TerminationStatusCode

end

@kwdef mutable struct DualWarmStart

	linking_values::Matrix{Float64}
	const strategy::String = "global"
	epsilon::Float64 = 0.001

end

function define_restricted_master_problem(
	gurobi_env,
	crew_route_data::CrewRouteData,
	crew_avail_ixs,
	fire_plan_data::FirePlanData,
	fire_avail_ixs,
	dual_warm_start::Union{Nothing, DualWarmStart} = nothing,
)

	# get dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_route_data)

	# inititalze JuMP model
	m = Model(() -> Gurobi.Optimizer(gurobi_env))
	set_optimizer_attribute(m, "OptimalityTol", 1e-9)
	set_optimizer_attribute(m, "FeasibilityTol", 1e-9)


	# decision variables for crew routes and fire plans
	@variable(m, route[c = 1:num_crews, r ∈ crew_avail_ixs[c]] >= 0)
	@variable(m, plan[g = 1:num_fires, p ∈ fire_avail_ixs[g]] >= 0)

	if ~isnothing(dual_warm_start)

		# dual stabilization variables
		@variable(m, delta_plus[g = 1:num_fires, t = 1:num_time_periods] >= 0)
		@variable(m, delta_minus[g = 1:num_fires, t = 1:num_time_periods] >= 0)

	end

	# constraints that you must choose a plan per crew and per fire
	@constraint(m, route_per_crew[c = 1:num_crews],
		sum(route[c, r] for r ∈ 1:crew_avail_ixs[c]) == 1)
	@constraint(m, plan_per_fire[g = 1:num_fires],
		sum(plan[g, p] for p ∈ fire_avail_ixs[g]) >= 1)

	# linking constraint
	if isnothing(dual_warm_start)

		# for each fire and time period
		@constraint(m, linking[g = 1:num_fires, t = 1:num_time_periods],

			# crews at fire
			sum(
				route[c, r] * crew_route_data.fires_fought[c, r, g, t]
				for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
			)
			>=

			# crews suppressing
			sum(
				plan[g, p] * fire_plan_data.crews_present[g, p, t]
				for p ∈ fire_avail_ixs[g]
			))

	elseif dual_warm_start.strategy == "global"

		# get expected dual value ratios
		ratios = dual_warm_start.linking_values
		ratios = ratios / sum(ratios)

		@constraint(m, linking[g = 1:num_fires, t = 1:num_time_periods],

			# crews at fire
			sum(
				route[c, r] * crew_route_data.fires_fought[c, r, g, t]
				for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
			)
			+

			# perturbation
			delta_plus[g, t] - delta_minus[g, t] -
			sum(ratios .* delta_plus) + sum(ratios .* delta_minus)
			>=

			# crews suppressing
			sum(
				plan[g, p] * fire_plan_data.crews_present[g, p, t]
				for p ∈ fire_avail_ixs[g]
			))

		# this constrant neutralizes the perturbation, will be presolved away if RHS is 0
		# but raising the RHS slightly above 0 allows the perturbation
		@constraint(m, perturb[g = 1:num_fires, t = 1:num_time_periods],
			delta_plus[g, t] + delta_minus[g, t] <= 0)
	else
		error("Dual stabilization type not implemented")
	end


	@objective(m, Min,

		# route costs
		sum(
			route[c, r] * route_data.route_costs[c, r]
			for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
		)
		+

		# suppression plan costs
		sum(
			plan[g, p] * fire_plan_data.plan_costs[g, p]
			for g ∈ 1:num_fires, p ∈ fire_avail_ixs[g]
		)
	)

	return RestrictedMasterProblem(
		m,
		route,
		plan,
		route_per_crew,
		plan_per_fire,
		linking,
		perturb,
		MOI.OPTIMIZE_NOT_CALLED,
	)

end


function add_column_to_plan_data!(
	plan_data::FirePlanData,
	fire::Int64,
	cost::Float64,
	crew_demands::Array{Int16, 2},
)

	# add 1 to number of plans for this fire, store the index
	plan_data.plans_per_fire[fire] += 1
	ix = plan_data.plans_per_fire[fire]

	# append the route cost
	plan_data.plan_costs[fire, ix] = cost

	# append the fires fought
	plan_data.crews_present[fire, ix, :] = crew_demands

end

function add_column_to_route_data!(
	route_data::CrewRouteData,
	crew::Int64,
	cost::Float64,
	fires_fought::BitArray{2},
)

	# add 1 to number of routes for this crew, store the index
	route_data.routes_per_crew[crew] += 1
	ix = route_data.routes_per_crew[crew]

	# append the route cost
	route_data.route_costs[crew, ix] = cost

	# append the fires fought
	route_data.fires_fought[crew, ix, :, :] = fires_fought

end

function add_column_to_master_problem!(
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	crew::Int16,
	ix::Int64,
)

    # define variable
	rmp.routes[crew, ix] =
		@variable(rmp.model, base_name = "route[$crew,$ix]", lower_bound = 0)

	# update coefficient in objective
	set_objective_coefficient(
		rmp.model,
		rmp.routes[crew, ix],
		crew_routes.route_costs[crew, ix],
	)

	# update coefficient in constraints
	set_normalized_coefficient(rmp.route_per_crew[crew], rmp.routes[crew, ix], 1)
	set_normalized_coefficient.(
		rmp.supply_demand_linking,
		rmp.routes[crew, ix],
		crew_routes.fires_fought[crew, ix, :, :],
	)

end

function add_column_to_master_problem!(rmp::RestrictedMasterProblem, fire_plans::FirePlanData, fire::Int16, new_plan_ix)

    mp["plan"][fire, ix] = @variable(mp["m"], base_name = "plan[$fire,$ix]", lower_bound = 0)

    # update coefficient in objective
    set_objective_coefficient(mp["m"], mp["plan"][fire, ix], supp_plan_data.plan_costs[fire, ix])

    # update coefficient in constraints
    set_normalized_coefficient(mp["pi"][fire], mp["plan"][fire, ix], 1)
    set_normalized_coefficient.(mp["rho"][fire, :], mp["plan"][fire, ix],
        -supp_plan_data.crews_present[fire, ix, :])
end

function double_column_generation(
	rmp::RestrictedMasterProblem,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	crew_branching_rules::Vector{CrewSupplyBranchingRule},
	fire_branching_rules::Vector{FireDemandBranchingRule},
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	improving_column_relative_tolerance::Float64 = 1e-4)

	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes)

	# initialize with an (infeasible) dual solution that will suppress minimally
	fire_duals = zeros(num_fires) .+ Inf
	crew_duals = zeros(num_crews)
	linking_duals = zeros(num_fires, num_time_periods) .+ 1e30

	# initialize column generation loop
	terminate::Bool = false
	iteration = 0

	while ~terminate

		iteration += 1

		# default is to terminate unless we find a new variable
		new_column_found = false

		# for each crew
		for crew in 1:num_crews

			# extract the subproblem
			subproblem = crew_subproblems[crew]

			# generate the local costs of the arcs
			rel_costs, prohibited_arcs = get_adjusted_crew_arc_costs(
				subproblem.crew_arc_array_long,
				linking_duals,
				crew_branching_rules,
			)
			arc_costs = rel_costs .+ subproblem.arc_costs

			# solve the subproblem
			objective, arcs_used = crew_dp_subproblem(
				subproblem.crew_arc_array_wide,
				arc_costs,
				prohibited_arcs,
				subproblem.state_in_arcs,
			)

			# if there is an improving route
			if objective / crew_duals[crew] < 1 - improving_column_relative_tolerance

				# get the real cost, unadjusted for duals
				cost = sum(subproblem.arc_costs[arcs_used])

				# get the indicator matrix of fires fought at each time
				fires_fought = get_fires_fought(
					subproblem.crew_arc_array_wide,
					arcs_used,
					(num_fires, num_time_periods),
				)

				# add the route to the routes
				new_route_ix =
					add_column_to_route_data!(crew_routes, crew, cost, fires_fought)

				# update the master problem
				add_column_to_master_problem!(rmp, crew_routes, crew, new_route_ix)

				new_column_found = true
			end
		end

		# for each fire
		for fire in 1:num_fires

			# extract the subproblem
			subproblem = fire_subproblems[fire]

			# generate the local costs of the arcs
			rel_costs, prohibited_arcs = get_adjusted_fire_arc_costs(
				subproblem.fire_arc_array_long,
				linking_duals,
				fire_branching_rules,
			)
			arc_costs = rel_costs .+ subproblem.arc_costs

			# solve the subproblem
			objective, arcs_used = fire_dp_subproblem(
				subproblem.fire_arc_array_wide,
				arc_costs,
				prohibited_arcs,
				subproblem.state_in_arcs,
			)

			# if there is an improving plan
			if objective / fire_duals[fire] < 1 - improving_column_relative_tolerance

				# get the real cost, unadjusted for duals
				cost = sum(subproblem.arc_costs[arcs_used])

				# get the vector of crew demands at each time
				crew_demands = get_crew_demands(
					subproblem.fire_arc_array_wide,
					arcs_used,
					num_time_periods,
				)

				# add the plan to the plans
				new_plan_ix =
					add_column_to_plan_data!(fire_plans, fire, cost, crew_demands)

				# update the master problem
				add_column_to_master_problem!(rmp, fire_plans, fire, new_plan_ix)

				new_column_found = true
			end
		end

		# if we added at least one column, or we have not yet solved the restricted master problem, solve it
		if new_column_found | (iteration == 1)

			# TODO dual warm start passed in here
			optimize!(rmp.model)

			# if the master problem is infeasible (this can only happen on iteration 1, return infeasible)
			if (termination_status(rmp.model) == MOI.INFEASIBLE) |
			   (termination_status(rmp.model) == MOI.INFEASIBLE_OR_UNBOUNDED)

				@assert iteration == 1
				println("RMP solution infeasible, surprising to catch this here")
				rmp.termination_status = MOI.INFEASIBLE
				return rmp
			end

			# otherwise we have proof of optimality
		else
			rmp.termination_status = MOI.OPTIMAL
		end

	end
end