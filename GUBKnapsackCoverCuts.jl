
include("DoubleColumnGeneration.jl")

using IterTools

function get_gub_fire_relevant_suppression_data(
	fire::Int64,
	time::Int64,
	fire_plans::FirePlanData,
	rmp::RestrictedMasterProblem,
	used_plans,
)
	sample = fire_plans.crews_present[fire, used_plans[fire], :]
	values = [value(rmp.plans[fire, i]) for i in used_plans[fire]]
	coeff_order = sortperm(sample[:, time])
	allot = 0
	weight = 0
	pairs = Tuple[]
	for coeff in coeff_order
		if sample[coeff, time] > allot
			push!(pairs, (sample[coeff, time], weight))
			allot = sample[coeff, time]
			# weight = 0
		end
		weight += values[coeff]
	end
	return pairs
end

function get_crew_suppression_cdf_by_fire_and_time(
	crew_routes::CrewRouteData,
	t::Int64,
	used_routes,
	rmp::RestrictedMasterProblem,
)
	num_crews, _, num_fires, _ = size(crew_routes.fires_fought)

	cdf = zeros(num_crews, num_fires)
	for c in 1:num_crews
		all_fires_fought = crew_routes.fires_fought[c, used_routes[c], :, t]
		coeffs = [value(rmp.routes[c, i]) for i in used_routes[c]]
		fires_fought = all_fires_fought' * coeffs
		cdf[c, :] = fires_fought
	end

	@debug "after populate" cdf

	for col in eachcol(cdf)
		sort!(col)
	end

	@debug "after sort" cdf

	cdf = cumsum(1 .- cdf, dims = 1)

	return cdf
end


function get_gub_crew_relevant_routing_data(
	c::Int64,
	t::Int64,
	crew_routes::CrewRouteData,
	used_routes,
	rmp::RestrictedMasterProblem,
)
	sample =
		.~dropdims(
			maximum(crew_routes.fires_fought[c, used_routes[c], :, :], dims = 2),
			dims = 2,
		)
	values = [value(rmp.routes[c, i]) for i in used_routes[c]]
	coeff_order = sortperm(sample[:, t])
	allot = 0
	weight = 0
	pairs = Tuple[]
	for coeff in coeff_order
		if sample[coeff, t] > allot
			push!(pairs, (sample[coeff, t], weight))
			allot = sample[coeff, t]
			# weight = 0
		end
		weight += values[coeff]
	end
	return pairs
end

function enumerate_minimal_cuts(
	crew_allots::Vector{Vector{Tuple}},
	fire_allots::Vector{Vector{Tuple}},
	cut_search_limit_per_time::Int64)

	all_cuts = []

	# find the crews that are inactive at time t in all used routes
	inactive_crews = Int64[]
	for crew in eachindex(crew_allots)
		if (length(crew_allots[crew]) > 0)
			if (crew_allots[crew][1][1] == true) & (crew_allots[crew][1][2] == 0)
				push!(inactive_crews, crew)
			end
		end
	end

	# for now, assume that these crews were inactive for a "good" reason and force those to be part of any cover
	available_for_fires = length(crew_allots) - length(inactive_crews)

	### get all minimal GUB covers under this assumption ###

	# enumerate indices of all possible GUB covers 
	for i in eachindex(fire_allots)
		fire_allots[i] = vcat([(0, 0.0)], fire_allots[i])
	end
	allotment_option_ixs = []
	for i in eachindex(fire_allots)
		push!(allotment_option_ixs, [])
		for j in eachindex(fire_allots[i])
			push!(allotment_option_ixs[i], j)
		end
	end

	loop_count = 0
	for i in product(allotment_option_ixs...)
		loop_count += 1
		if loop_count >= cut_search_limit_per_time
			if loop_count > 1
				@info "Broke cover cut enumeration early" loop_count
			end
			break
		end
		allot = 0
		cost = 0
		min_allot = 10000
		for j in eachindex(i)
			cur_allot = fire_allots[j][i[j]][1]
			allot += cur_allot
			cost += fire_allots[j][i[j]][2]
			if i[j] > 1
				min_allot = min(min_allot, cur_allot)
			end
		end
		if (cost < 1 - 0.01) & (allot > available_for_fires) &
		   (allot - min_allot <= available_for_fires)
			cut_allotments = [fire_allots[j][i[j]][1] for j in eachindex(i)]
			push!(
				all_cuts,
				(available_for_fires, cost, cut_allotments, inactive_crews),
			)
		end
	end

	return all_cuts
end

function adjust_cut_fire_allotment(
	current_allotment::Vector{Int64},
	incumbent_weighted_average::Vector{Float64},
	budget::Int64,
)

	adjusted_allotment = copy(current_allotment)
	while sum(adjusted_allotment) > budget + 1
		max_diff = -1000.0
		max_diff_ix = -1
		for ix in eachindex(adjusted_allotment)
			if (adjusted_allotment[ix] > 0) &
			   (adjusted_allotment[ix] - incumbent_weighted_average[ix] > max_diff)
				max_diff_ix = ix
				max_diff = adjusted_allotment[ix] - incumbent_weighted_average[ix]
			end
		end
		adjusted_allotment[max_diff_ix] -= 1
	end
	return adjusted_allotment
end

function extract_usages(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	rmp::RestrictedMasterProblem,
)
	## TODO refactor with copied code below ##

	# get problem dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	all_fire_allots = Array{Vector{Tuple}, 2}(undef, num_fires, num_time_periods)
	all_crew_allots = Array{Vector{Tuple}, 2}(undef, num_crews, num_time_periods)

	# get the plans that are used in the current RMP solution
	used_plans = []
	for g in 1:num_fires
		plans = [
			i[1] for
			i in eachindex(rmp.plans[g, :]) if value(rmp.plans[g, i...]) > 1e-4
		]
		push!(used_plans, plans)
	end

	# get the routes that are used in the current RMP solution
	used_routes = []
	for c in 1:num_crews
		routes = [
			i[1] for
			i in eachindex(rmp.routes[c, :]) if value(rmp.routes[c, i...]) > 1e-4
		]
		push!(used_routes, routes)
	end

	# for each time period
	for t in 1:num_time_periods

		# process these routes and plans into GUB-relevant allotments (needed for dominated cuts due to GUB constraints)
		for f ∈ 1:num_fires
			all_fire_allots[f, t] = get_gub_fire_relevant_suppression_data(
				f,
				t,
				fire_plans,
				rmp,
				used_plans,
			)
		end

		for c ∈ 1:num_crews
			all_crew_allots[c, t] = get_gub_crew_relevant_routing_data(
				c,
				t,
				crew_routes,
				used_routes,
				rmp,
			)
		end
	end

	return all_fire_allots, all_crew_allots
end

function find_single_fire_knapsack_cuts(
	fire_allots,
	cdf_crew_assign,
	argsort_crew_assign,
)

	num_crews, num_fires, num_time_periods = size(cdf_crew_assign)

	cuts = []

	# get the potential cuts per fire, time, demand
	cut_violations = copy(cdf_crew_assign)
	cut_violations = 1 .- cdf_crew_assign
	for t ∈ 1:num_time_periods
		for g ∈ 1:num_fires
			for c ∈ 1:num_crews
				for (ix, (allot, survival_f)) ∈ enumerate(fire_allots[g, t])
					if allot > num_crews - c
						lifted_coeff = min(c, allot + c - num_crews)
						if ix == length(fire_allots[g, t])
							fire_demand_weight = 1 - survival_f
						else
							next_survival_f = fire_allots[g, t][ix+1][2]
							fire_demand_weight = next_survival_f - survival_f
						end
						cut_violations[c, g, t] += lifted_coeff * fire_demand_weight
					end
				end
			end
		end
	end

	# choose the max violation per fire, time
	max_per_fire = mapslices(findmax, cut_violations, dims = 1)
	for t ∈ 1:num_time_periods
		for g ∈ 1:num_fires
			if max_per_fire[1, g, t][1] > 1.01
				n_crews_in_cut = max_per_fire[1, g, t][2]
				crews_present = argsort_crew_assign[1:n_crews_in_cut, g, t]
				cut_allotments =
					[ix == g ? num_crews - n_crews_in_cut + 1 : 0 for ix ∈ 1:num_fires]
				push!(
					cuts,
					(
						num_crews - n_crews_in_cut,
						0,
						cut_allotments,
						crews_present,
						t,
					),
				)
			end
		end
	end

	@info "cut_violations" cdf_crew_assign cut_violations cuts


	return cuts
end
function find_knapsack_cuts(
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	rmp::RestrictedMasterProblem,
	cut_search_limit_per_time::Int64,
	general_cuts::Bool,
	single_fire_cuts::Bool,
)

	# get problem dimensions
	num_crews, _, num_fires, num_time_periods = size(crew_routes.fires_fought)

	all_fire_allots = Array{Vector{Tuple}, 2}(undef, num_fires, num_time_periods)
	all_crew_allots = Array{Vector{Tuple}, 2}(undef, num_crews, num_time_periods)

	# get the plans that are used in the current RMP solution
	used_plans = []
	for g in 1:num_fires
		plans = [
			i[1] for
			i in eachindex(rmp.plans[g, :]) if value(rmp.plans[g, i...]) > 1e-4
		]
		push!(used_plans, plans)
	end

	# get the routes that are used in the current RMP solution
	used_routes = []
	for c in 1:num_crews
		routes = [
			i[1] for
			i in eachindex(rmp.routes[c, :]) if value(rmp.routes[c, i...]) > 1e-4
		]
		push!(used_routes, routes)
	end

	# get the current fire suppression relaxed solution
	fire_alloc, _ =
		get_fire_and_crew_incumbent_weighted_average(rmp, crew_routes, fire_plans)

	# intitialize output 
	knapsack_gub_cuts = []

	# for each time period
	for t in 1:num_time_periods

		# process these routes and plans into GUB-relevant allotments (needed for dominated cuts due to GUB constraints)
		for f ∈ 1:num_fires
			all_fire_allots[f, t] = get_gub_fire_relevant_suppression_data(
				f,
				t,
				fire_plans,
				rmp,
				used_plans,
			)
		end
		for c ∈ 1:num_crews
			all_crew_allots[c, t] = get_gub_crew_relevant_routing_data(
				c,
				t,
				crew_routes,
				used_routes,
				rmp,
			)
		end

	end

	if single_fire_cuts
		crew_assign = get_crew_incumbent_weighted_average(rmp, crew_routes)
		@info "crew_assign" crew_assign
		cdf_crew_assign =
			mapslices(cumsum, mapslices(sort, crew_assign, dims = 1), dims = 1)
		argsort_crew_assign = mapslices(sortperm, crew_assign, dims = 1)
		cuts = find_single_fire_knapsack_cuts(
			all_fire_allots,
			cdf_crew_assign,
			argsort_crew_assign,
		)
		for cut in cuts
			orig_fire_allots = cut[3]
			unused_crews = cut[4]
			fire_allot_dict = Dict(
				ix => orig_fire_allots[ix] for
				ix in eachindex(orig_fire_allots) if orig_fire_allots[ix] > 0
			)
			num_crews_in_cut = length(unused_crews)

			gub_cut = GUBCoverCut(
				cut[5],
				fire_allot_dict,
				unused_crews,
				length(keys(fire_allot_dict)) + num_crews_in_cut - 1,
			)
			push!(knapsack_gub_cuts, gub_cut)
		end
	end

	if general_cuts
		for t in 1:num_time_periods

			minimal_cuts =
				enumerate_minimal_cuts(
					all_crew_allots[:, t],
					all_fire_allots[:, t],
					cut_search_limit_per_time,
				)


			for cut in minimal_cuts
				orig_fire_allots = cut[3]
				unused_crews = cut[4]
				adjusted_cut =
					adjust_cut_fire_allotment(
						orig_fire_allots,
						fire_alloc[:, t],
						cut[1],
					)
				fire_allot_dict = Dict(
					ix => adjusted_cut[ix] for
					ix in eachindex(orig_fire_allots) if orig_fire_allots[ix] > 0
				)
				num_crews_in_cut = length(unused_crews)

				gub_cut = GUBCoverCut(
					t,
					fire_allot_dict,
					unused_crews,
					length(keys(fire_allot_dict)) + num_crews_in_cut - 1,
				)
				push!(knapsack_gub_cuts, gub_cut)
			end
		end
	end

	return knapsack_gub_cuts
end


function incorporate_gub_cover_cuts_into_fire_subproblem!(
	out_dict,
	cuts,
	fire,
	submodel,
)

	# for each cut ix
	for ix in eachindex(cuts)

		# if the cut ix is not in the cut dict
		if ix ∉ keys(out_dict)

			# grab the cut and initialize a lookup dictionary arc_ix => cut coeff
			cut = cuts[ix]
			costs = Dict{Int64, Int8}()

			# if the fire is involved in the cut 
			if fire in keys(cut.fire_lower_bounds)

				# for each arc in the subproblem
				for i in 1:length(submodel.arc_costs)

					# if it considers the given time
					if submodel.long_arcs[i, FM.TIME_FROM] == cut.time_ix

						# find the excess allotment 
						exceeds_cut_allotment_by =
							submodel.long_arcs[i, FM.CREWS_PRESENT] -
							cut.fire_lower_bounds[fire][1]

						# if this arc meets the allotment
						if exceeds_cut_allotment_by >= 0

							costs[i] = Float64(cut.fire_lower_bounds[fire][2])
							if length(cut.fire_lower_bounds) == 1

								# based on the number of crews involved in the cut, we can lift
								num_crews_in_cut = length(cut.inactive_crews)
								costs[i] +=
									min(exceeds_cut_allotment_by, num_crews_in_cut)
							end
						end
					end
				end
			end
			out_dict[ix] = costs
		end
	end
	return out_dict
end


function incorporate_gub_cover_cuts_into_crew_subproblem!(
	out_dict,
	cuts,
	crew,
	submodel,
)

	for ix in eachindex(cuts)
		if ix ∉ keys(out_dict)
			cut = cuts[ix]
			costs = Dict{Int64, Int8}()
			if crew in cut.inactive_crews
				ixs = findall(submodel.long_arcs[:, CM.CREW_NUMBER] .== crew)
				for i in ixs
					# -1 for any that do suppress at the time
					if (submodel.long_arcs[i, CM.TIME_TO] == cut.time_ix) &
					   (submodel.long_arcs[i, CM.TO_TYPE] == CM.FIRE_CODE)

						# the route needs to suppress a fire in the cut to avoid the penalty
						fires = [i for i in keys(cut.fire_lower_bounds)]
						if (submodel.long_arcs[i, CM.LOC_TO] ∈ fires)
							costs[i] = -1
						end
					end
				end
			end
			out_dict[ix] = costs
		end
	end
end

function update_cut_fire_mp_lookup!(
	cut_fire_mp_lookup::Dict{Tuple{Int64, Int64}, Int8},
	cut::GUBCoverCut,
	fire_plans::FirePlanData,
	fire::Int64,
	plan_ix::Int64,
)
	exceeds_cut_allotment_by =
		fire_plans.crews_present[fire, plan_ix..., cut.time_ix] -
		cut.fire_lower_bounds[fire][1]

	# if this arc meets the allotment
	if exceeds_cut_allotment_by >= 0

		cut_fire_mp_lookup[(fire, plan_ix)] = cut.fire_lower_bounds[fire][2]
		if length(cut.fire_lower_bounds) == 1

			# based on the number of crews involved in the cut, we can lift
			num_crews_in_cut = length(cut.inactive_crews)
			cut_fire_mp_lookup[(fire, plan_ix)] +=
				min(exceeds_cut_allotment_by, num_crews_in_cut)
		end

		return true
	end

	return false

end

function push_cut_to_rmp!!(
	rmp::RestrictedMasterProblem,
	cut_data::GUBCoverCutData,
	cut::GUBCoverCut,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
)

	cut_time = cut.time_ix
	cut_crew_allots = cut.inactive_crews
	cut_rhs = cut.rhs

	ix = cut_data.cuts_per_time[cut_time] + 1

	cut_fire_mp_lookup = Dict{Tuple{Int64, Int64}, Int8}()
	for f in keys(cut.fire_lower_bounds)
		for i in eachindex(rmp.plans[f, :])
			update_cut_fire_mp_lookup!(cut_fire_mp_lookup, cut, fire_plans, f, i...)
		end
	end

	# TODO allow coeffs for this cut too
	cut_crew_mp_lookup = Dict{Tuple{Int64, Int64}, Int8}()
	for c in cut_crew_allots
		for i in eachindex(rmp.routes[c, :])

			# crew has to suppress one of the fires in the cut to avoid entering this cut
			fires = [i for i in keys(cut.fire_lower_bounds)]
			if maximum(crew_routes.fires_fought[c, i..., fires, cut_time]) == 0
				cut_crew_mp_lookup[(c, i...)] = 1
			end
		end
	end

	lhs = []
	for ix in eachindex(cut_fire_mp_lookup)
		push!(lhs, cut_fire_mp_lookup[ix] * rmp.plans[ix])
	end

	for ix in eachindex(cut_crew_mp_lookup)
		push!(lhs, cut_crew_mp_lookup[ix] * rmp.routes[ix])
	end

	# update data structures
	rmp.gub_cover_cuts[cut_time, ix] = @constraint(rmp.model, -sum(lhs) >= -cut_rhs)
	cut_data.cut_dict[(cut_time, ix)] = cut
	cut_data.cuts_per_time[cut_time] = ix
	cut_data.crew_mp_lookup[(cut_time, ix)] = cut_crew_mp_lookup
	cut_data.fire_mp_lookup[(cut_time, ix)] = cut_fire_mp_lookup
end

function find_and_incorporate_knapsack_gub_cuts!!(
	gub_cut_data::GUBCoverCutData,
	cut_search_limit_per_time::Int64,
	rmp::RestrictedMasterProblem,
	crew_routes::CrewRouteData,
	fire_plans::FirePlanData,
	crew_models,
	fire_models;
	general_cuts=true,
	single_fire_cuts=false
)

	num_crews, _, num_fires, _ = size(crew_routes.fires_fought)
	num_cuts_found = 0
	keep_iterating = true
	while keep_iterating
		knapsack_cuts = find_knapsack_cuts(
			crew_routes,
			fire_plans,
			rmp,
			cut_search_limit_per_time,
			general_cuts,
			single_fire_cuts
		)
		num_cuts_found += length(knapsack_cuts)

		## lots of choices for how many times to introduce cuts and re-optimize
		## can look for more after reoptimizing to cut off new solution
		## but maybe don't care if the new incumbent is nowhere near the eventual solution
		## for now let's do one round of minimal cuts but could do all of them or tune based on change in objective

		keep_iterating = false

		# if length(knapsack_cuts) == 0
		#     keep_iterating = false
		# end

		for cut in knapsack_cuts
			push_cut_to_rmp!!(
				rmp,
				gub_cut_data,
				cut,
				crew_routes,
				fire_plans,
			)
		end

		optimize!(rmp.model)
	end

	for fire ∈ 1:num_fires
		incorporate_gub_cover_cuts_into_fire_subproblem!(
			gub_cut_data.fire_sp_lookup[fire],
			gub_cut_data.cut_dict,
			fire,
			fire_models[fire],
		)
	end

	for crew ∈ 1:num_crews
		incorporate_gub_cover_cuts_into_crew_subproblem!(
			gub_cut_data.crew_sp_lookup[crew],
			gub_cut_data.cut_dict,
			crew,
			crew_models[crew],
		)
	end

	return num_cuts_found
end
