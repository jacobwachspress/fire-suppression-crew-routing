using JuMP

struct TimeSpaceNetwork # TODO always make time the first index

	arc_costs::Vector{Float64}
	state_in_arcs::Array{Vector{Int64}}
	state_out_arcs::Array{Vector{Int64}}
	model_type::String

	# not quite sure how much this helps performance, but storing 2 copies
	# of array data allows always column-wise access
	long_arcs::Matrix{Int64}
	wide_arcs::Matrix{Int64}

    # avoid later allocations with modified arc costs and prohibited arcs
    modified_arc_costs::Vector{Float64}
    prohibited_arcs::BitVector

	# precalculate arc lookups to supply-demand linking variables
	supply_demand_dual_arc_lookup::Matrix{Vector{Int64}}

	# hack for fire models
	start_time_period::Union{Int64, Nothing}

end

struct CrewRouteData

	n_crews::Int64
	routes_per_crew::Vector{Int64}
	route_costs::Matrix{Float64}
	fires_fought::BitArray{4}
	arcs_used::Matrix{Vector{Int64}}

end

function CrewRouteData(
	max_routes::Int64,
	num_fires::Int64,
	num_crews::Int64,
	num_time_periods::Int64,
)

	return CrewRouteData(num_crews, zeros(num_crews),
		Matrix{Float64}(undef, num_crews, max_routes),
		BitArray(undef, num_crews, max_routes, num_fires, num_time_periods) .> 2,
		Matrix{Vector{Int64}}(undef, num_crews, max_routes),
	)
end

struct FirePlanData

	n_fires::Int64
	plans_per_fire::Vector{Int64}
	plan_costs::Matrix{Float64}
	crews_present::Array{Int64, 3}
	arcs_used::Matrix{Vector{Int64}}

end

function FirePlanData(
	max_supp_plans::Int64,
	num_fires::Int64,
	num_time_periods::Int64,
)

	return FirePlanData(num_fires, zeros(num_fires),
		Matrix{Float64}(undef, num_fires, max_supp_plans),
		zeros(Int64, (num_fires, max_supp_plans, num_time_periods)),
		Matrix{Vector{Int64}}(undef, num_fires, max_supp_plans),
	)
end

struct RobustCut
	time_ix::Int64
	fire_coeffs::Dict{Int64, Vector{Float64}}
	crew_coeffs::Vector{Float64}
	rhs::Union{Int64, Float64}
end


struct CutData

	cut_dict::Dict{Any, RobustCut}
	cuts_per_time::Vector{Int64}
	fire_sp_lookup::Dict{Int64, Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}}
	crew_sp_lookup::Dict{Int64, Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}}
	fire_mp_lookup::Dict{Tuple{Int64, Int64}, Dict{Tuple{Int64, Int64}, Float64}}
	crew_mp_lookup::Dict{Tuple{Int64, Int64}, Dict{Tuple{Int64, Int64}, Float64}}
end

# constructor
function CutData(num_crews::Int, num_fires::Int, num_time_periods::Int)

	cut_objects = Dict{Any, RobustCut}()
	cuts_per_time = [0 for t ∈ 1:num_time_periods]
	fire_cut_sp_lookup =
		Dict{Int64, Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}}()
	crew_cut_sp_lookup =
		Dict{Int64, Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}}()
	for fire ∈ 1:num_fires
		fire_cut_sp_lookup[fire] = Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}()
	end
	for crew ∈ 1:num_crews
		crew_cut_sp_lookup[crew] = Dict{Tuple{Int64, Int64}, Dict{Int64, Float64}}()
	end

	return CutData(
		cut_objects,
		cuts_per_time,
		fire_cut_sp_lookup,
		crew_cut_sp_lookup,
		Dict{Tuple{Int64, Int64}, Dict{Tuple{Int64, Int64}, Int64}}(),
		Dict{Tuple{Int64, Int64}, Dict{Tuple{Int64, Int64}, Int64}}(),
	)
end

function first_cut_dominates(cut_1::RobustCut, cut_2::RobustCut)

	if cut_1.time_ix != cut_2.time_ix
		return false
	end

	for fire in keys(cut_2.fire_coeffs)
		if (fire ∉ keys(cut_1.fire_coeffs)) || (any(
			(cut_2.fire_coeffs[fire] ./ cut_2.rhs) .>
			(cut_1.fire_coeffs[fire] ./ cut_1.rhs),
		))
			return false
		end
	end
	if any((cut_2.crew_coeffs ./ cut_2.rhs) .> (cut_1.crew_coeffs ./ cut_1.rhs))
		return false
	end

	return true
end


function restrict_CutData(orig::CutData, ixs)
	cut_objects = Dict(a => b for (a, b) in orig.cut_dict if a in ixs)
	cuts_per_time = orig.cuts_per_time
	fire_sp_lookup = Dict(
		a => Dict(b => c for (b, c) in orig.fire_sp_lookup[a] if b ∈ ixs) for
		a ∈ keys(orig.fire_sp_lookup)
	)
	crew_sp_lookup = Dict(
		a => Dict(b => c for (b, c) in orig.crew_sp_lookup[a] if b ∈ ixs) for
		a ∈ keys(orig.crew_sp_lookup)
	)
	fire_mp_lookup = Dict(a => b for (a, b) in orig.fire_mp_lookup if a ∈ ixs)
	crew_mp_lookup = Dict(a => b for (a, b) in orig.crew_mp_lookup if a ∈ ixs)
	return CutData(
		cut_objects,
		cuts_per_time,
		fire_sp_lookup,
		crew_sp_lookup,
		fire_mp_lookup,
		crew_mp_lookup)
end


mutable struct RestrictedMasterProblem # TODO can make some JuMP things const?

	# model
	const model::JuMP.Model

	# fast lookups for variables
	crew_column_ixs::Vector{Vector{Int}}
	fire_column_ixs::Vector{Vector{Int}}

	# vars
	const routes::JuMP.Containers.SparseAxisArray # could speed up?
	const plans::JuMP.Containers.SparseAxisArray # could speed up?
    const deferred_num_crews::JuMP.Containers.DenseAxisArray

	# constraints
	const route_per_crew::Vector{ConstraintRef}
	const plan_per_fire::Vector{ConstraintRef}
	const supply_demand_linking::Matrix{ConstraintRef}
	const gub_cover_cuts::JuMP.Containers.SparseAxisArray
	const fire_allotment_branches::Vector{ConstraintRef}
	# linking_perturbation::Matrix{ConstraintRef}

	# termination status
	termination_status::MOI.TerminationStatusCode

end
