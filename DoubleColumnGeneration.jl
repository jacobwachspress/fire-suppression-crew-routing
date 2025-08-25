include("CommonStructs.jl")
include("Subproblems.jl")
include("BranchingRules.jl")

using JuMP


@kwdef mutable struct DualWarmStart

	linking_values::Matrix{Float64}
	const strategy::String = "global"
	epsilon::Float64 = 0.001

end

"""
Perform double column generation to solve the (relaxed) restricted master problem `rmp`
by iteratively adding columns to model feasible fire suppression plans and crew routes.
This function optimizes suppression plans and routes simultaneously to satisfy
supply-demand constraints and minimize costs.

# Arguments

  - `rmp::RestrictedMasterProblem` (modified): The restricted master problem instance representing the optimization problem.
  - `crew_routes::CrewRouteData` (modified): Data structure containing information (cost, assignments) about all generated crew routes.
  - `fire_plans::FirePlanData` (modified): Data structure containing information (cost, demands) about all generated fire plans.
  - `cut_data::CutData` (modified): Data structure containing information about cuts, including lookups to incorporate coeffs into master problem and subproblems.
  - `crew_subproblems::Vector{TimeSpaceNetwork}`: Vector of data structures containing all static info about crew subproblems.
  - `fire_subproblems::Vector{TimeSpaceNetwork}`: Vector of data structures containing all static info about fire subproblems.
  - `crew_branching_rules::Vector{CrewAssignmentBranchingRule}`: Vector of branching rules that indicate whether crew j suppresses fire g at time t.
  - `fire_branching_rules::Vector{FireDemandBranchingRule}`: Vector of branching rules that indicate whether fire g demands <=d crews or >d crews at time t.
  - `global_fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule}`: Vector of branching rules that may place a cap on demand across all fires and times
  - `upper_bound::Float64`: Global upper bound given by the best feasible solution we have to the integer problem.
  - `timing::Bool`: Flag for whether we will track and return detailed timings
  - `improving_column_abs_tolerance::Float64`: Absolute improvement needed (using reduced cost bound) to add a column (default: 1e-10).
  - `local_gap_rel_tolerance::Float64`: Relative tolerance (using Lagrangian bound) needed to accept solution as optimal (default: 1e-9).
"""
function double_column_generation!!!!(
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	cut_data::CutData,
	crew_subproblems::Vector{TimeSpaceNetwork},
	fire_subproblems::Vector{TimeSpaceNetwork},
	crew_branching_rules::Vector{CrewAssignmentBranchingRule},
	fire_branching_rules::Vector{FireDemandBranchingRule},
	global_fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule},
	fires_to_ignore::Vector{Int64};
	upper_bound::Float64,
	timing::Bool,
	time_limit::Float64 = Inf,
	improving_column_abs_tolerance::Float64 = 1e-10,
	local_gap_rel_tolerance::Float64 = 1e-5)

	# initialize timing dictionary
	details = Dict{String, Float64}()
	details["master_problem"] = 0.0
	details["fire_subproblems"] = 0.0
	details["crew_subproblems"] = 0.0
	t = time()

	# gather global information
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	if rmp.termination_status == MOI.OPTIMIZE_NOT_CALLED
		# initialize with an (infeasible) dual solution that will suppress minimally
		fire_duals = zeros(num_fires) .+ Inf
		crew_duals = zeros(num_crews)
		linking_duals = zeros(num_fires, num_time_periods) .+ 1e30
		cut_duals = normalized_rhs.(rmp.gub_cover_cuts) .* 0
		global_fire_allot_duals = normalized_rhs.(rmp.fire_allotment_branches) .* 0
	else
		fire_duals = dual.(rmp.plan_per_fire)
		crew_duals = dual.(rmp.route_per_crew)
		linking_duals = dual.(rmp.supply_demand_linking)
		cut_duals = dual.(rmp.gub_cover_cuts)
		global_fire_allot_duals = dual.(rmp.fire_allotment_branches)
	end

	ub = copy(upper_bound)

	# initialize column generation loop
	continue_iterating::Bool = true
	iteration = 0

	# add in dummy plans for the fires_to_ignore
	for fire in fires_to_ignore
		add_column_to_master_problem!!(
			rmp,
			cut_data,
			fire_plans,
			global_fire_allotment_branching_rules,
			fire,
			1,
		)
	end

	while continue_iterating

		iteration += 1
		reduced_cost_sum = 0

		if timing
			t = time()
		end

		crew_objectives = zeros(Float64, num_crews)
		crew_arcs_used = [Int[] for crew ∈ 1:num_crews]

		# for each crew
		
		Threads.@threads for crew in 1:num_crews

			crew_subproblems[crew].prohibited_arcs .&= false
			for i ∈ eachindex(crew_subproblems[crew].modified_arc_costs)
				crew_subproblems[crew].modified_arc_costs[i] = crew_subproblems[crew].arc_costs[i]
			end
			adjust_crew_arc_costs!!(
				crew_subproblems[crew].modified_arc_costs,
				crew_subproblems[crew].prohibited_arcs,
				crew,
				linking_duals,
				crew_subproblems[crew].supply_demand_dual_arc_lookup,
				crew_branching_rules,
			)

			# adjust the arc costs for the cuts
			cut_adjust_arc_costs!(
				crew_subproblems[crew].modified_arc_costs,
				cut_data.crew_sp_lookup[crew],
				cut_duals,
			)

			# solve the subproblem
			objective, arcs_used = crew_dp_subproblem(
				crew_subproblems[crew].wide_arcs,
				crew_subproblems[crew].modified_arc_costs,
				crew_subproblems[crew].prohibited_arcs,
				crew_subproblems[crew].state_in_arcs,
			)

			# adjust the objective for the cuts (we gave - coeff if allotment not broken, give + coeff here)
			for (ix, cut) in cut_data.cut_dict
				if cut.crew_coeffs[crew] > 1e-20
					objective += (cut.crew_coeffs[crew] * cut_duals[ix])
				end
			end

			crew_objectives[crew] = objective
			crew_arcs_used[crew] = arcs_used
		end
		

		for crew ∈ 1:num_crews

			objective = crew_objectives[crew]
			arcs_used = crew_arcs_used[crew]

			# if there is an improving route
			if objective < crew_duals[crew] - improving_column_abs_tolerance

				reduced_cost_sum += (objective - crew_duals[crew])

				# get the real cost, unadjusted for duals
				cost = sum(crew_subproblems[crew].arc_costs[arcs_used])

				# get the indicator matrix of fires fought at each time
				fires_fought = get_fires_fought(
					crew_subproblems[crew].wide_arcs,
					arcs_used,
					(num_fires, num_time_periods),
				)

				@debug "crew route" crew fires_fought

				# add the route to the routes
				new_route_ix =
					add_column_to_route_data!(crew_routes, crew, cost, fires_fought, arcs_used)

				# update the master problem
				add_column_to_master_problem!!(
					rmp,
					cut_data,
					crew_routes,
					crew,
					new_route_ix,
				)
			end
		end

		if timing
			details["crew_subproblems"] += time() - t
			t = time()
		end

		fire_objectives = zeros(Float64, num_fires)
		fire_arcs_used = [Int[] for fire ∈ 1:num_fires]

		# for each fire
		Threads.@threads for fire in [g for g ∈ 1:num_fires if g ∉ fires_to_ignore]

			# generate the local costs of the arcs
			fire_subproblems[fire].prohibited_arcs .&= false
			for i ∈ eachindex(fire_subproblems[fire].modified_arc_costs)
				fire_subproblems[fire].modified_arc_costs[i] = fire_subproblems[fire].arc_costs[i]
			end
			adjust_fire_arc_costs!!(
				fire_subproblems[fire].modified_arc_costs,
				fire_subproblems[fire].prohibited_arcs,
				fire,
				fire_subproblems[fire].supply_demand_dual_arc_lookup,
				fire_subproblems[fire].long_arcs,
				linking_duals[fire, :],
				[rule for rule ∈ fire_branching_rules if rule.fire_ix == fire],
			)

			# adjust the arc costs for the cuts
			cut_adjust_arc_costs!(
				fire_subproblems[fire].modified_arc_costs,
				cut_data.fire_sp_lookup[fire],
				cut_duals,
			)

			# for each branching rule
			for ix in eachindex(global_fire_allotment_branching_rules)

				rule = global_fire_allotment_branching_rules[ix]

				# if this is a <= 0 rule, we have more prohibited arcs
				if ~rule.geq_flag
					for arc_ix ∈ rule.fire_sp_arc_lookup[fire]
						fire_subproblems[fire].prohibited_arcs[arc_ix] = true
					end
				end

				# we need to do a proper dual adjustment
				adjust_fire_sp_arc_costs!(
					fire_subproblems[fire].modified_arc_costs,
					rule,
					fire,
					global_fire_allot_duals[ix],
				)
			end

			# solve the subproblem
			objective, arcs_used = fire_dp_subproblem(
				fire_subproblems[fire].wide_arcs,
				fire_subproblems[fire].modified_arc_costs,
				fire_subproblems[fire].prohibited_arcs,
				fire_subproblems[fire].state_in_arcs,
			)

			fire_objectives[fire] = objective
			fire_arcs_used[fire] = arcs_used
		end
		


		for fire ∈ [fire for fire ∈ 1:num_fires if fire ∉ fires_to_ignore]

			objective = fire_objectives[fire]
			arcs_used = fire_arcs_used[fire]

			# if there is an improving plan
			if objective < fire_duals[fire] - improving_column_abs_tolerance

				reduced_cost_sum += (objective - fire_duals[fire])

				# get the real cost, unadjusted for duals
				cost = sum(fire_subproblems[fire].arc_costs[arcs_used])

				# get the vector of crew demands at each time
				crew_demands = get_crew_demands(
					fire_subproblems[fire].wide_arcs,
					arcs_used,
					num_time_periods,
				)

				# add the plan to the plans
				new_plan_ix =
					add_column_to_plan_data!(fire_plans, fire, cost, crew_demands, arcs_used)

				@debug "fire plan" fire crew_demands
				# update the master problem
				add_column_to_master_problem!!(
					rmp,
					cut_data,
					fire_plans,
					global_fire_allotment_branching_rules,
					fire,
					new_plan_ix,
				)
			end
		end



		if timing
			details["fire_subproblems"] += time() - t
		end

		@debug "total reduced cost" reduced_cost_sum ub reduced_cost_sum / ub local_gap_rel_tolerance
		continue_iterating =
			((iteration == 1) || (-reduced_cost_sum / ub > local_gap_rel_tolerance)) && (time_limit > time() - t)

		# if we have not found columns with enough reduced cost
		if ~continue_iterating

			# because of discretization, we do not actually have a guarantee that
			# the deferral variables are 0. set them to 0 here and keep iterating.
			# the deferral variables still help a lot with convergence.
			# alternative is to accept the solution with deferrals, implicitly
			# improving discretization... too complicated but should be a better solution
			if maximum(value.(rmp.deferred_num_crews)) > 1e-5
				for g ∈ 1:num_fires
					for t ∈ 1:num_time_periods
						fix(rmp.deferred_num_crews[g, t], 0, force = true)
					end
				end
				continue_iterating = true
				@info "remove deferrals" iteration
			end
		end

		if continue_iterating

			# TODO dual warm start passed in here
			if timing
				t = time()
			end
			optimize!(rmp.model)
			if timing
				details["master_problem"] += (time() - t)
			end

			if termination_status(rmp.model) != MOI.OPTIMAL
				@info "non optimal termination status" termination_status(rmp.model)
			end


			## TODO FIX THIS LOGIC AND INFEASIBLE LOGIC
			rmp.termination_status = MOI.ITERATION_LIMIT

			# grab dual values (or farkas vector if infeasible)
			fire_duals = dual.(rmp.plan_per_fire)
			crew_duals = dual.(rmp.route_per_crew)
			linking_duals = dual.(rmp.supply_demand_linking)
			cut_duals = dual.(rmp.gub_cover_cuts)
			global_fire_allot_duals = dual.(rmp.fire_allotment_branches)

			# if the master problem is infeasible
			if (termination_status(rmp.model) == MOI.INFEASIBLE) |
			   (termination_status(rmp.model) == MOI.INFEASIBLE_OR_UNBOUNDED)

				# log this
				@debug "RMP is infeasible, using dual certificate to get dual values" iteration fire_duals crew_duals linking_duals cut_duals dual_status(
					rmp.model,
				)

				# set status
				rmp.termination_status = MOI.INFEASIBLE

				# scale dual values to a feasible dual solution with cost "upper_bound"
				# (linking_duals omitted because 0 RHS)
				dual_costs = 0
				for ix in eachindex(fire_duals)
					dual_costs +=
						(normalized_rhs(rmp.plan_per_fire[ix]) * fire_duals[ix])
				end

				for ix in eachindex(crew_duals)
					dual_costs +=
						(normalized_rhs(rmp.route_per_crew[ix]) * crew_duals[ix])
				end

				for ix in eachindex(cut_duals)
					dual_costs +=
						(normalized_rhs(rmp.gub_cover_cuts[ix]) * cut_duals[ix])
				end

				for ix in eachindex(global_fire_allot_duals)
					dual_costs +=
						(
							normalized_rhs(rmp.fire_allotment_branches[ix]) *
							global_fire_allot_duals[ix]
						)
				end

				scale = upper_bound / dual_costs
				fire_duals = fire_duals .* scale
				crew_duals = crew_duals .* scale
				linking_duals = linking_duals .* scale
				cut_duals = cut_duals .* scale
				global_fire_allot_duals = global_fire_allot_duals .* scale
			else
				ub = objective_value(rmp.model)
				lb = ub + reduced_cost_sum
				if lb > upper_bound
					@debug "prune by bound" lb upper_bound iteration
					continue_iterating = false
					rmp.termination_status = MOI.OBJECTIVE_LIMIT
				end
			end

			supp = get_fire_incumbent_weighted_average(
				rmp,
				fire_plans,
				num_fires,
				num_time_periods,
			)
			@debug "progress" iteration linking_duals supp


		# if no new column added, we have proof of optimality
		else
			# re-optimze for JuMP reasons (access attrs) just in case we added a column 
			# but then stopped due to too small reduced cost improvement
			if timing
				t = time()
			end
			optimize!(rmp.model)

			if (termination_status(rmp.model) == MOI.INFEASIBLE) |
				(termination_status(rmp.model) == MOI.INFEASIBLE_OR_UNBOUNDED)
 
				 # log this
				 @debug "prune by infeasibility"
				 rmp.termination_status = MOI.INFEASIBLE

			else

				rmp.termination_status = MOI.LOCALLY_SOLVED
				@debug "end DCG" iteration termination_status(rmp.model) objective_value(
					rmp.model,
				)

			end

			if timing
				details["master_problem"] += (time() - t)
			end

		end
	end

	details["iteration"] = Float64(iteration)
	return details
end



function define_restricted_master_problem(
	gurobi_env,
	crew_route_data::CrewRouteData,
	crew_avail_ixs::Vector{Vector{Int64}},
	fire_plan_data::FirePlanData,
	fire_avail_ixs::Vector{Vector{Int64}},
	cut_data::CutData,
	fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule},
	deferral_stabilization::Bool,
	fires_to_ignore::Vector{Int64},
	dual_warm_start::Union{Nothing, DualWarmStart} = nothing,
)
	@info "Define restricted master problem" fires_to_ignore

	# get dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_route_data.fires_fought)

	# inititalze JuMP model
	m = direct_model(Gurobi.Optimizer(gurobi_env))
	set_optimizer_attribute(m, "OutputFlag", 0) # put this first so others don't print
	set_optimizer_attribute(m, "OptimalityTol", 1e-9)
	set_optimizer_attribute(m, "FeasibilityTol", 1e-9)
	set_optimizer_attribute(m, "InfUnbdInfo", 1)
	# set_optimizer_attribute(m, "OutputFlag", 1)


	# decision variables for crew routes and fire plans
	@variable(m, route[c = 1:num_crews, r ∈ crew_avail_ixs[c]] >= 0)
	@variable(m, plan[g = 1:num_fires, p ∈ fire_avail_ixs[g]] >= 0)

	# add deferral stabilization variables
	@variable(
		m,
		deferred_num_crews[
			g = 1:num_fires,
			t = 0:num_time_periods,
		] >= 0
	)
	for g ∈ 1:num_fires
		fix(deferred_num_crews[g, 0], 0, force = true)
	end

	# fix deferral stabilization variables to 0 if not stabilizing
	if ~deferral_stabilization
		for g ∈ 1:num_fires
			for t ∈ 1:num_time_periods
				fix(deferred_num_crews[g, t], 0, force = true)
			end
		end
	end

	if ~isnothing(dual_warm_start)

		error("Not implemented")
		# dual stabilization variables
		# @variable(m, delta_plus[g = 1:num_fires, t = 1:num_time_periods] >= 0)
		# @variable(m, delta_minus[g = 1:num_fires, t = 1:num_time_periods] >= 0)

	end

	# constraints that you must choose a plan per crew and per fire
	@constraint(m, route_per_crew[c = 1:num_crews],
		sum(route[c, r] for r ∈ crew_avail_ixs[c]) == 1)
	@constraint(m, plan_per_fire[g = 1:num_fires],
		sum(plan[g, p] for p ∈ fire_avail_ixs[g]) >= 1)

	## constraints for cuts

	# get proper coefficients of columns in each cut
	cut_ixs = keys(cut_data.cut_dict)

	fire_plan_ixs = Dict()
	fire_plan_coeffs = Dict()
	fire_lookup = cut_data.fire_mp_lookup
	for cut_ix in cut_ixs
		cut_plan_ixs = []
		cut_plan_coeffs = []
		for (fire_ix, coeff) in fire_lookup[cut_ix]
			if fire_ix in eachindex(plan)
				push!(cut_plan_ixs, fire_ix)
				push!(cut_plan_coeffs, -coeff)
			end
		end
		fire_plan_ixs[cut_ix] = cut_plan_ixs
		fire_plan_coeffs[cut_ix] = cut_plan_coeffs
	end

	crew_route_ixs = Dict()
	crew_route_coeffs = Dict()
	crew_lookup = cut_data.crew_mp_lookup
	for cut_ix in cut_ixs
		cut_route_ixs = []
		cut_route_coeffs = []
		for (crew_ix, coeff) in crew_lookup[cut_ix]
			if crew_ix in eachindex(route)
				push!(cut_route_ixs, crew_ix)
				push!(cut_route_coeffs, -coeff)
			end
		end
		crew_route_ixs[cut_ix] = cut_route_ixs
		crew_route_coeffs[cut_ix] = cut_route_coeffs
	end

	# need it to default to SparseAxisArray when empty, maybe there is a better way
	@constraint(
		m,
		gub_cover_cuts[
			t = 1:num_time_periods,
			u = 1:10000;
			(t, u) ∈ keys(cut_data.cut_dict),
		],
		sum(
			crew_route_coeffs[t, u][i] * route[crew_route_ixs[t, u][i]] for
			i ∈ eachindex(crew_route_coeffs[t, u])
		) +
		sum(
			fire_plan_coeffs[t, u][i] * plan[fire_plan_ixs[t, u][i]] for
			i ∈ eachindex(fire_plan_coeffs[t, u])
		) >=
		-cut_data.cut_dict[(t, u)].rhs
	)

	# container for fire allotment branching rules
	@constraint(
		m,
		fire_allotment_branches[eachindex(fire_allotment_branching_rules)],
		0 >= 0
	)

	# set constraint RHS and plan coeffs
	for ix in eachindex(fire_allotment_branching_rules)
		rule = fire_allotment_branching_rules[ix]

		# (following >= convention)
		sign = 2 * Int(rule.geq_flag) - 1

		# possible RHS values are <= 0, >= 1
		set_normalized_rhs(fire_allotment_branches[ix], Int(rule.geq_flag))

		# get coeff for any fire plan exceeding the allotment
		for (fire_plan_ix, excess) ∈ rule.mp_lookup
			if fire_plan_ix ∈ eachindex(plan)
				set_normalized_coefficient(
					fire_allotment_branches[ix],
					plan[fire_plan_ix],
					sign * excess,
				)
			end
		end

	end

	# linking constraint
	if isnothing(dual_warm_start)

		# for each fire and time period
		@constraint(m, linking[g = 1:num_fires, t = 1:num_time_periods],

			# crews at fire
			sum(
				route[c, r] * crew_route_data.fires_fought[c, r, g, t]
				for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
			) +
			deferred_num_crews[g, t-1] - deferred_num_crews[g, t]
			>=

			# crews suppressing
			sum(
				plan[g, p] * fire_plan_data.crews_present[g, p, t]
				for p ∈ fire_avail_ixs[g]
			))

	elseif dual_warm_start.strategy == "global"

		error("Not implemented")

		# # get expected dual value ratios
		# ratios = dual_warm_start.linking_values
		# ratios = ratios / sum(ratios)

		# @constraint(m, linking[g = 1:num_fires, t = 1:num_time_periods],

		# 	# crews at fire
		# 	sum(
		# 		route[c, r] * crew_route_data.fires_fought[c, r, g, t]
		# 		for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
		# 	)
		# 	+

		# 	# perturbation
		# 	delta_plus[g, t] - delta_minus[g, t] -
		# 	sum(ratios .* delta_plus) + sum(ratios .* delta_minus)
		# 	>=

		# 	# crews suppressing
		# 	sum(
		# 		plan[g, p] * fire_plan_data.crews_present[g, p, t]
		# 		for p ∈ fire_avail_ixs[g]
		# 	))

		# # this constrant neutralizes the perturbation, will be presolved away if RHS is 0
		# # but raising the RHS slightly above 0 allows the perturbation
		# @constraint(m, perturb[g = 1:num_fires, t = 1:num_time_periods],
		# 	delta_plus[g, t] + delta_minus[g, t] <= 0)
	else
		error("Dual stabilization type not implemented")
	end


	@objective(m, Min,

		# route costs
		sum(
			route[c, r] * crew_route_data.route_costs[c, r]
			for c ∈ 1:num_crews, r ∈ crew_avail_ixs[c]
		)
		+

		# suppression plan costs
		sum(
			plan[g, p] * fire_plan_data.plan_costs[g, p]
			for g ∈ 1:num_fires, p ∈ fire_avail_ixs[g]
		)

		- 

		# ignore fires that are not started yet
		sum(
			plan[g, p] * fire_plan_data.plan_costs[g, p] 
			for g ∈ fires_to_ignore, p ∈ fire_avail_ixs[g]
		)
	)

	return RestrictedMasterProblem(
		m,
		crew_avail_ixs,
		fire_avail_ixs,
		route,
		plan,
		deferred_num_crews,
		route_per_crew,
		plan_per_fire,
		linking,
		gub_cover_cuts,
		fire_allotment_branches,
		MOI.OPTIMIZE_NOT_CALLED,
	)

end

"""
Add a new suppression plan to the FirePlanData structure for a specific fire.

# Arguments

  - `plan_data::FirePlanData` (modified): The FirePlanData structure to which the column will be added.
  - `fire::Int64`: The index of the fire for which the plan is being added.
  - `cost::Float64`: The cost associated with the plan for the given fire.
  - `crew_demands::Vector{Int64}`: The crews required for the plan at each time.

# Returns

  - `ix::Int64`: The index of the newly added column in the plan data for the specified fire.

# Description

  - Increments the count of plans for the specified fire.
  - Appends the plan cost to the plan costs matrix for the specified fire.
  - Appends the crew demands to the crews present matrix for the specified fire.
"""
function add_column_to_plan_data!(
	plan_data::FirePlanData,
	fire::Int64,
	cost::Float64,
	crew_demands::Vector{Int64},
	arcs_used::Vector{Int64},
)
	# add 1 to number of plans for this fire, store the index
	plan_data.plans_per_fire[fire] += 1
	ix = plan_data.plans_per_fire[fire]

	# append the route cost
	plan_data.plan_costs[fire, ix] = cost

	# append the fires fought
	plan_data.crews_present[fire, ix, :] = crew_demands

	# append the arcs used
	plan_data.arcs_used[fire, ix] = arcs_used

	return ix

end

"""
Add a new route to the CrewRouteData structure for a specific crew.

# Arguments

  - `route_data::CrewRouteData` (modified): The CrewRouteData structure to which the column will be added.
  - `crew::Int64`: The index of the crew for which the route data is being added.
  - `cost::Float64`: The cost associated with the route for the given crew.
  - `fires_fought::BitArray{2}`: A 2-dimensional BitArray representing the fires fought by the crew at each time for the given route.

# Returns

  - `ix::Int64`: The index of the newly added column in the route data for the specified crew.

# Description

  - Increments the count of routes for the specified crew.
  - Appends the route cost to the route costs matrix for the specified crew.
  - Appends the fires fought to the fires_fought array for the specified crew.
"""
function add_column_to_route_data!(
	route_data::CrewRouteData,
	crew::Int64,
	cost::Float64,
	fires_fought::BitArray{2},
	arcs_used::Vector{Int64},
)

	# add 1 to number of routes for this crew, store the index
	route_data.routes_per_crew[crew] += 1
	ix = route_data.routes_per_crew[crew]

	# append the route cost
	route_data.route_costs[crew, ix] = cost

	# append the fires fought
	route_data.fires_fought[crew, ix, :, :] = fires_fought

	# append the arcs used
	route_data.arcs_used[crew, ix] = arcs_used

	return ix

end

"""
Add a new column to the RestrictedMasterProblem representing a crew route.

# Arguments

  - `rmp::RestrictedMasterProblem` (modified): The RestrictedMasterProblem object to which the column will be added.
  - `cut_data::CutData` (modified): The CutData object containing information about the cuts.
  - `crew_routes::CrewRouteData`: The CrewRouteData object containing information about crew routes.
  - `crew::Int64`: The index of the crew for which the route is being added.
  - `ix::Int64`: The index (for this crew) of the route being added.

# Description

  - Defines a variable representing the new route in the RestrictedMasterProblem.
  - Updates the coefficient of this route in the objective function of the RestrictedMasterProblem.
  - Updates coefficient of this route in all relevant constraints.
"""
function add_column_to_master_problem!!(
	rmp::RestrictedMasterProblem,
	cut_data::CutData,
	crew_routes::CrewRouteData,
	crew::Int64,
	ix::Int64,
)

	# define variable
	rmp.routes[crew, ix] =
		@variable(rmp.model, base_name = "route[$crew,$ix]", lower_bound = 0)

	# update index lookup
	push!(rmp.crew_column_ixs[crew], ix)

	# update coefficient in objective
	set_objective_coefficient(
		rmp.model,
		rmp.routes[crew, ix],
		crew_routes.route_costs[crew, ix],
	)

	## update coefficient in constraints

	# route per crew
	set_normalized_coefficient(rmp.route_per_crew[crew], rmp.routes[crew, ix], 1)

	# supply demand linking
	set_normalized_coefficient.(
		rmp.supply_demand_linking,
		rmp.routes[crew, ix],
		crew_routes.fires_fought[crew, ix, :, :],
	)

	# cuts

	# for each cut in the cut data
	for (cut_ix, cut) ∈ cut_data.cut_dict

		# if this crew is involved in the cut
		if cut.crew_coeffs[crew] > 1e-20

			# get the fires not suppressed at the given time in order to enter cut
			fires = [i for i in keys(cut.fire_coeffs)]

			# if these fires are not suppressed at the given time
			if maximum(crew_routes.fires_fought[crew, ix, fires, cut.time_ix]) == 0

				# set the coefficient of this route in this cut to -1 (negated because >=)
				set_normalized_coefficient(
					rmp.gub_cover_cuts[cut_ix],
					rmp.routes[crew, ix],
					-1,
				)

				# add the plan to the cut mp lookup
				cut_data.crew_mp_lookup[cut_ix][(crew, ix)] = 1

			end

		end
	end

end

"""
Add a new column to the RestrictedMasterProblem representing a fire plan.

# Arguments

  - `rmp::RestrictedMasterProblem` (modified): The RestrictedMasterProblem object to which the column will be added.
  - `cut_data::CutData` (modified): The CutData object containing information about the cuts.
  - `fire_plans::FirePlanData`: The FirePlanData object containing information about fire plans.
  - `fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule}`: The vector of GlobalFireAllotmentBranchingRule containing information about global fire allotment branching rules.
  - `fire::Int64`: The index of the fire for which the plan is being added.
  - `ix::Int64`: The index (for this fire)of the plan being added.

# Description

  - Defines a variable representing the new plan in the RestrictedMasterProblem.
  - Updates the coefficient in the objective function of the RestrictedMasterProblem.
  - Updates coefficient of this plan in all relevant constraints.
"""
function add_column_to_master_problem!!(
	rmp::RestrictedMasterProblem,
	cut_data::CutData,
	fire_plans::FirePlanData,
	fire_allotment_branching_rules::Vector{GlobalFireAllotmentBranchingRule},
	fire::Int64,
	ix::Int64,
)

	# define variable
	rmp.plans[fire, ix] =
		@variable(rmp.model, base_name = "plan[$fire,$ix]", lower_bound = 0)

	# update index lookup
	push!(rmp.fire_column_ixs[fire], ix)

	# update coefficient in objective
	set_objective_coefficient(
		rmp.model,
		rmp.plans[fire, ix],
		fire_plans.plan_costs[fire, ix],
	)

	# update coefficient in constraints

	# plan per fire
	set_normalized_coefficient(rmp.plan_per_fire[fire], rmp.plans[fire, ix], 1)

	# supply demand linking
	set_normalized_coefficient.(
		rmp.supply_demand_linking[fire, :],
		rmp.plans[fire, ix],
		-fire_plans.crews_present[fire, ix, :],
	)

	# cuts

	# for each cut in the cut data
	for (cut_ix, cut) ∈ cut_data.cut_dict

		# if this fire is involved in the cut
		if fire ∈ keys(cut.fire_coeffs)

			# update mp lookup and see if this plan has >0 coeff in the cut
			plan_in_cut = update_cut_fire_mp_lookup!(
				cut_data.fire_mp_lookup[cut_ix],
				cut,
				fire_plans,
				fire,
				ix,
			)

			if plan_in_cut
				# set the coefficient of this plan in this cut to the coefficient of the cut (negated because >=)
				set_normalized_coefficient(
					rmp.gub_cover_cuts[cut_ix],
					rmp.plans[fire, ix],
					-cut_data.fire_mp_lookup[cut_ix][(fire, ix)],
				)
			end

		end
	end

	# global fire allotment branching rule, add new plan to mp_lookup
	for rule_ix in eachindex(rmp.fire_allotment_branches)
		rule = fire_allotment_branching_rules[rule_ix]
		sign = 2 * Int(rule.geq_flag) - 1

		# if fire plan exceeds the allotment
		if (fire, ix) ∈ keys(rule.mp_lookup)
			set_normalized_coefficient(
				rmp.fire_allotment_branches[rule_ix],
				rmp.plans[(fire, ix)],
				sign * rule.mp_lookup[(fire, ix)],
			)
		end
	end
end

function get_fire_incumbent_weighted_average(
	rmp::RestrictedMasterProblem,
	fire_plans::FirePlanData,
	num_fires::Int,
	num_time_periods::Int,
)

	fire_allotment = zeros(num_fires, num_time_periods)
	for ix in eachindex(rmp.plans)
		if value(rmp.plans[ix]) > 0
			fire_allotment[ix[1], :] +=
				fire_plans.crews_present[ix..., :] * value(rmp.plans[ix])
		end
	end

	return fire_allotment

end

function get_crew_incumbent_weighted_average(
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
)

	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	crew_allotment = zeros(Float64, num_crews, num_fires, num_time_periods)
	for crew in 1:num_crews
		for col in rmp.crew_column_ixs[crew]
			weight = value(rmp.routes[(crew, col)])
			if weight > 0
				crew_allotment[crew, :, :] +=
					crew_routes.fires_fought[crew, col, :, :] * weight
			end
		end
	end

	return crew_allotment
end

function get_fire_and_crew_incumbent_weighted_average(
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
)

	# get problem dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	fire_allotment = get_fire_incumbent_weighted_average(
		rmp,
		fire_plans,
		num_fires,
		num_time_periods,
	)
	crew_allotment = get_crew_incumbent_weighted_average(rmp, crew_routes)
	return fire_allotment, crew_allotment
end

function get_cost_due_to_fires_and_crews(
	solved_rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
)

	# get problem dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	# get the cost due to fires
	fire_cost = 0
	for ix in eachindex(solved_rmp.plans)
		if value(solved_rmp.plans[ix]) > 0
			f = ix[1]
			plan = ix[2]
			fire_cost += value(solved_rmp.plans[ix]) * fire_plans.plan_costs[f, plan]
		end
	end

	# get the cost due to crews
	crew_cost = 0
	for ix in eachindex(solved_rmp.routes)
		if value(solved_rmp.routes[ix]) > 0
			c = ix[1]
			route = ix[2]
			crew_cost += value(solved_rmp.routes[ix]) * crew_routes.route_costs[c, route]
		end
	end
			

	return fire_cost, crew_cost
end

function get_fire_and_crew_arcs_used(
	solved_rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
)

	# get problem dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	# get the arcs used by fires
	fire_arcs_used = Vector{Vector{Int64}}(undef, num_fires)
	for f in 1:num_fires
		fire_arcs_used[f] = Int64[]
	end
	for ix in eachindex(solved_rmp.plans)
		if value(solved_rmp.plans[ix]) > 0.99
			f = ix[1]
			plan = ix[2]
			fire_arcs_used[f] = fire_plans.arcs_used[f, plan]
		end
	end

	# get the arcs used by crews
	crew_arcs_used = Vector{Vector{Int64}}(undef, num_crews)
	for c in 1:num_crews
		crew_arcs_used[c] = Int64[]
	end
	for ix in eachindex(solved_rmp.routes)
		if value(solved_rmp.routes[ix]) > 0.99
			c = ix[1]
			route = ix[2]
			crew_arcs_used[c] = crew_routes.arcs_used[c, route]
		end
	end

	return fire_arcs_used, crew_arcs_used
end