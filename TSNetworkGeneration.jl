include("CommonStructs.jl")
using DataFrames, CSV, DelimitedFiles

module CrewArcArrayIndices

# indices for each crew arc
CREW_NUMBER = 1
FROM_TYPE = 2
LOC_FROM = 3
TO_TYPE = 4
LOC_TO = 5
TIME_FROM = 6
TIME_TO = 7
REST_FROM = 8
REST_TO = 9

# integer lookup for "FIRE" and "BASE"
FIRE_CODE = 1
BASE_CODE = 2

end

module FireArcArrayIndices

# indices for each crew arc
STATE_FROM = 2
TIME_FROM = 3
TIME_TO = 4
STATE_TO = 5
CREWS_PRESENT = 6

end

const CM = CrewArcArrayIndices
const FM = FireArcArrayIndices

struct LocationAndRestStatus

    rest_by::Vector{Int64}
    current_fire::Vector{Int64}
    rested_periods::Vector{Int64}

end

struct DistancesAndTravelTimes

    ff_dist::Matrix{Float64}
    bf_dist::Matrix{Float64}
    ff_tau::Matrix{Int64}
    bf_tau::Matrix{Int64}

end


@kwdef struct KeyArcIndices

    # fire flow data
    f_out::Array{Vector{Int64}}
    f_in::Array{Vector{Int64}}

    # base flow data
    b_out::Array{Vector{Int64}}
    b_in::Array{Vector{Int64}}

    # total crews suppressing each fire
    supp_fire::Array{Vector{Int64}}

    # start constraints
    start::Array{Vector{Int64}}

    # assignments out of region
    out_of_region::Union{Nothing,Array{Vector{Int64}}} = nothing

end

function generate_arcs(
    dists_and_times::DistancesAndTravelTimes,
    crew_status::LocationAndRestStatus,
    num_crews::Int64,
    num_fires::Int64,
    num_time_periods::Int64,
    break_length::Int64=2,
)

    # get fire-to-fire arcs
    ff = [
        [
            c,
            CM.FIRE_CODE,
            f_from,
            CM.FIRE_CODE,
            f_to,
            t_from,
            t_from + dists_and_times.ff_tau[f_to, f_from],
            rest,
            rest,
        ]
        for c ∈ 1:num_crews, f_from ∈ 1:num_fires, f_to ∈ 1:num_fires,
        t_from ∈ 1:num_time_periods, rest ∈ 0:1
    ]
    ff = copy(reduce(hcat, ff)')

    # get fire-to-fire arcs from start, based on current crew locations
    from_start_ff = [
        [
            c,
            CM.FIRE_CODE,
            crew_status.current_fire[c],
            CM.FIRE_CODE,
            f_to,
            0,
            dists_and_times.ff_tau[f_to, crew_status.current_fire[c]],
            0,
            0,
        ]
        for
        c ∈ 1:num_crews, f_to ∈ 1:num_fires if crew_status.current_fire[c] != -1
    ]
    from_start_ff = copy(reduce(hcat, from_start_ff)')

    # get base-to-fire arcs
    rf = [
        [
            c,
            CM.BASE_CODE,
            c,
            CM.FIRE_CODE,
            f_to,
            t_from,
            t_from + dists_and_times.bf_tau[c, f_to],
            rest,
            rest,
        ]
        for c ∈ 1:num_crews, f_to ∈ 1:num_fires, t_from ∈ 1:num_time_periods,
        rest ∈ 0:1
    ]
    rf = copy(reduce(hcat, rf)')

    # get base-to-fire arcs from start
    from_start_rf = [
        [c, CM.BASE_CODE, c, CM.FIRE_CODE, f_to, 0, dists_and_times.bf_tau[c, f_to], 0, 0]
        for
        c ∈ 1:num_crews, f_to ∈ 1:num_fires if crew_status.current_fire[c] == -1
    ]
    from_start_rf = copy(reduce(hcat, from_start_rf)')

    # get fire-to-base arcs
    fr = [
        [
            c,
            CM.FIRE_CODE,
            f_from,
            CM.BASE_CODE,
            c,
            t_from,
            t_from + dists_and_times.bf_tau[c, f_from],
            rest,
            rest,
        ]
        for c ∈ 1:num_crews, f_from ∈ 1:num_fires, t_from ∈ 1:num_time_periods,
        rest ∈ 0:1
    ]
    fr = copy(reduce(hcat, fr)')

    # get fire-to-base arcs from start, based on cs.current crew locations
    from_start_fr = [
        [
            c,
            CM.FIRE_CODE,
            crew_status.current_fire[c],
            CM.BASE_CODE,
            c,
            0,
            dists_and_times.bf_tau[c, crew_status.current_fire[c]],
            0,
            0,
        ]
        for c ∈ 1:num_crews if crew_status.current_fire[c] != -1
    ]
    from_start_fr = copy(reduce(hcat, from_start_fr)')

    # get base-to-base arcs
    rr = [
        [
            c,
            CM.BASE_CODE,
            c,
            CM.BASE_CODE,
            c,
            t_from,
            t_from + 1 + (break_length - 1) * rest,
            0,
            rest,
        ]
        for c ∈ 1:num_crews, t_from ∈ 1:num_time_periods, rest ∈ 0:1
    ]
    rr = copy(reduce(hcat, rr)')
    rr_rested = [
        [c, CM.BASE_CODE, c, CM.BASE_CODE, c, t_from, t_from + 1, 1, 1]
        for c ∈ 1:num_crews, t_from ∈ 1:num_time_periods
    ]
    rr_rested = copy(reduce(hcat, rr_rested)')

    # get base-to-base arcs from start, based on cs.current days rested
    from_start_rr = [
        [c, CM.BASE_CODE, c, CM.BASE_CODE, c, 0,
            1 + (break_length - max(crew_status.rested_periods[c], 0) - 1) * rest, 0,
            rest]
        for c ∈ 1:num_crews, rest ∈ 0:1 if crew_status.current_fire[c] == -1
    ]
    from_start_rr = copy(reduce(hcat, from_start_rr)')

    A = vcat(
        ff,
        from_start_ff,
        rf,
        from_start_rf,
        fr,
        from_start_fr,
        rr,
        rr_rested,
        from_start_rr,
    )

    return A
end

function get_distance(from_type, from_ix, to_type, to_ix, fire_fire, base_fire)

    dist = 0

    # if fire to fire
    if (from_type == CM.FIRE_CODE) & (to_type == CM.FIRE_CODE)
        dist = fire_fire[from_ix, to_ix]

        # if fire to base
    elseif (from_type == CM.FIRE_CODE) & (to_type == CM.BASE_CODE)
        dist = base_fire[to_ix, from_ix]

        # if base to fire
    elseif (from_type == CM.BASE_CODE) & (to_type == CM.FIRE_CODE)
        dist = base_fire[from_ix, to_ix]

        # otherwise dist still 0
    end

    return dist
end

function get_static_crew_arc_costs(gd, arcs, cost_param_dict)

    # get number of arcs
    n_arcs = length(arcs[:, 1])

    # initialize costs to 0
    costs = zeros(n_arcs)

    # if there is travel cost per mile
    if "cost_per_mile" in keys(cost_param_dict)

        # find the miles for each arc
        miles_per_arc = [
            get_distance(arcs[i, 2], arcs[i, 3],
                arcs[i, 4], arcs[i, 5],
                gd.ff_dist, gd.bf_dist) for i in 1:n_arcs
        ]
        # add to costs
        costs = costs .+ (cost_param_dict["cost_per_mile"] * miles_per_arc)
    end

    # if there are rest violations
    if "rest_violation" in keys(cost_param_dict)

        # find the rest violation scores
        rest_violation_matrix = cost_param_dict["rest_violation"]
        rest_violations = [
            (arcs[i, 8] == 0) & (arcs[i, 6] > 0) ?
            rest_violation_matrix[arcs[i, 1], arcs[i, 6]] : 0
            for i in 1:n_arcs
        ]

        # add to costs
        costs = costs .+ rest_violations
    end

    if "fight_fire" in keys(cost_param_dict)
        costs =
            costs .+ [
                (arcs[i, 4] == CM.FIRE_CODE) ? cost_param_dict["fight_fire"] : 0
                for i in 1:n_arcs
            ]
    end

    return copy(costs) ./ 1e6 # divide by 1e6 because Gurobi tolerance
end

function crew_data_from_path(path, travel_speed::Float64)

    # get distance from fire f to fire g 
    fire_dists = readdlm(path * "/fire_distances.csv", ',')

    # get distance from base c to fire g (NUM_CREWS-by-NUM_FIRES)
    base_fire_dists = readdlm(path * "/base_fire_distances.csv", ',')

    # initialize travel times (number of periods) from fire f to fire g
    tau = convert(Array{Int}, ones(size(fire_dists)))

    # initialize number of periods to travel from base c to fire g (NUM_CREWS-by-NUM_FIRES)
    tau_base_to_fire = convert(Array{Int}, ones((size(base_fire_dists))))

    # add travel times
    tau .+= Int.(round.(fire_dists ./ travel_speed))
    tau_base_to_fire .+= Int.(round.(base_fire_dists ./ travel_speed))

    @debug "travel times" tau tau_base_to_fire
    # read intial crew statuses (location, period by which they must rest)
    # (-1 in current_fire means crew is currently at base)
    # (rested_periods is the amount of time crew has been at base, relevant for completing rest)
    crew_starts = CSV.read(path * "/sample_crew_starts.csv", DataFrame)
    rest_by = crew_starts[!, "rest_by"]
    current_fire = crew_starts[!, "current_fire"]
    rested_periods = crew_starts[!, "rested_periods"]


    return (
        DistancesAndTravelTimes(fire_dists, base_fire_dists, tau, tau_base_to_fire),
        LocationAndRestStatus(rest_by, current_fire, rested_periods))
end

function define_network_constraint_data(arcs, num_crews, num_fires, num_time_periods)
    # just need in_arcs
    # fix numbers

    # shorten some variable names
    C = num_crews
    G = num_fires
    T = num_time_periods

    # get number of arcs
    n_arcs = length(arcs[:, 1])

    ## flow balance ##

    # initialize arrays of vectors for flow balance
    f_out = Array{Vector{Int64}}(undef, C, G, T, 2)
    f_in = Array{Vector{Int64}}(undef, C, G, T, 2)
    b_out = Array{Vector{Int64}}(undef, C, T, 2)
    b_in = Array{Vector{Int64}}(undef, C, T, 2)
    start = Array{Vector{Int64}}(undef, C)

    # for each crew
    for crew in 1:C

        # get indices of this crew's arcs only
        crew_ixs = [i for i in 1:n_arcs if arcs[i, 1] == crew]

        # get time 0 indices
        start[crew] = [i for i in crew_ixs if arcs[i, 6] == 0]

        # for each time period
        for tm in 1:T

            # for each rest state
            for rest in 1:2

                # get arcs leaving crew base at this time with this rest
                b_out[crew, tm, rest] = [
                    i for i in crew_ixs if
                    (arcs[i, 2] == CM.BASE_CODE) &
                    (arcs[i, 6] == tm) &
                    (arcs[i, 8] == rest - 1)
                ]

                # get arcs entering crew base at this time with this rest
                b_in[crew, tm, rest] = [
                    i for i in crew_ixs if
                    (arcs[i, 4] == CM.BASE_CODE) &
                    (arcs[i, 7] == tm) &
                    (arcs[i, 9] == rest - 1)
                ]
                # for each fire
                for fire in 1:G

                    # get arcs where this crew leaves this fire at this time
                    # with this rest state
                    f_out[crew, fire, tm, rest] = [
                        i for i in crew_ixs if
                        (arcs[i, 2] == CM.FIRE_CODE) &
                        (arcs[i, 3] == fire) &
                        (arcs[i, 6] == tm) &
                        (arcs[i, 8] == rest - 1)
                    ]

                    # get arcs where this crew enters this fire at this time
                    # with this rest state
                    f_in[crew, fire, tm, rest] = [
                        i for i in crew_ixs if
                        (arcs[i, 4] == CM.FIRE_CODE) &
                        (arcs[i, 5] == fire) &
                        (arcs[i, 7] == tm) &
                        (arcs[i, 9] == rest - 1)
                    ]
                end
            end
        end
    end

    ## linking constraints ##
    linking = Array{Vector{Int64}}(undef, G, T)
    for fire in 1:G
        for tm in 1:T

            # we count the crew as working *where they arrived* during this timestep
            linking[fire, tm] = [
                i for i in 1:n_arcs if (arcs[i, 4] == CM.FIRE_CODE) &
                (arcs[i, 5] == fire) &
                (arcs[i, 7] == tm)
            ]
        end
    end

    return KeyArcIndices(
        f_out=f_out,
        f_in=f_in,
        b_out=b_out,
        b_in=b_in,
        supp_fire=linking,
        start=start,
    )
end

function positive(x)
    return Int(x > 0)
end

function is_one(x)
    return Int(x == 1)
end

# should return matrix indexed by crew, time, 
function get_rest_penalties(
    num_crews,
    num_time_periods,
    rest_by_periods,
    lambda,
    accounting_func,
)

    penalties = zeros(num_crews, num_time_periods)

    for c in 1:num_crews
        penalties[c, :] = [
            lambda * accounting_func(t - rest_by_periods[c])
            for t in 1:num_time_periods
        ]
    end

    return penalties
end

function build_crew_models_from_empirical(
    num_crews::Int64, 
    num_fires::Int64, 
    num_time_periods::Int64,
    travel_speed::Float64
)

    # read in the selected fires
    fire_folder = "data/empirical_fire_models/raw/arc_arrays"
    selected_fires = CSV.read(fire_folder * "/" * "selected_fires.csv", DataFrame)   

    # read in the crew locations
    tau_base_to_fire = CSV.read(fire_folder * "/../" * "consolidated_distances.csv", DataFrame)

    # restrict to the fires whose fire_id is in the arc_file column of "selected_fires.csv"
    tau_base_to_fire = tau_base_to_fire[findall(in(selected_fires[:, "fire_id"]), tau_base_to_fire[:, "fire_id"]), :]
    
    # turn the tau_base_to_fire into a matrix, dropping the columns names and the fire_id column
    tau_base_to_fire = Matrix(tau_base_to_fire[:, 2:end])

    # transpose the matrix so that the rows are the crews and the columns are the fires
    tau_base_to_fire = copy(tau_base_to_fire')

    # minutes to days
    tau_base_to_fire = tau_base_to_fire / 60 / 24

    # round to nearest integer, minimum 1 day
    tau_base_to_fire = round.(Int, tau_base_to_fire) .+ 1

    # get the distance from the bases to the fires, assuming travel speed is miles per day
    base_fire_dists = tau_base_to_fire * travel_speed

    # initialize travel times (number of periods) from fire f to fire g
    tau = convert(Array{Int}, ones(num_fires, num_fires))

    # TODO fix fire-distances
    # for now they will all be the same, the mean of the base-fire distances
    fire_dists = zeros(num_fires, num_fires)
    dist = sum(base_fire_dists) / length(base_fire_dists)
    for i in 1:num_fires
        for j in 1:num_fires
            if i != j
                fire_dists[i, j] = dist
            end
        end
    end

    tau .+= Int.(round.(fire_dists ./ travel_speed))

    # TODO make better crew starts
    current_fire = [1, 2, 3, 1, 4, 5, 2, 3, -1, -1]
    current_fire = repeat(current_fire, 100)
    current_fire = current_fire[1:num_crews]
    rest_by = [15, 15, 10, 10, 8, 8, 8, 5, 5, 4]
    rest_by = repeat(rest_by, 100)
    rest_by = rest_by[1:num_crews]
    rested_periods = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1]
    rested_periods = repeat(rested_periods, 100)
    rested_periods = rested_periods[1:num_crews]

    crew_status = LocationAndRestStatus(rest_by, current_fire, rested_periods)
    dists_and_times = DistancesAndTravelTimes(fire_dists, base_fire_dists, tau, tau_base_to_fire)

    arcs = generate_arcs(
        dists_and_times,
        crew_status,
        num_crews,
        num_fires,
        num_time_periods,
    )

    rest_pen = get_rest_penalties(
        num_crews,
        num_time_periods,
        crew_status.rest_by,
        1e10,
        positive,
    )
    ALPHA = 200
    cost_params = Dict(
        "cost_per_mile" => 1,
        "rest_violation" => rest_pen,
        "fight_fire" => ALPHA,
    )

    crew_sps = TimeSpaceNetwork[]
    for crew in 1:num_crews

        n_arcs = length(arcs[:, 1])
        crew_arcs = arcs[[i for i in 1:n_arcs if arcs[i, CM.CREW_NUMBER] == crew], :]
        crew_wide_arcs = collect(crew_arcs')
        crew_arc_costs = get_static_crew_arc_costs(dists_and_times, crew_arcs, cost_params)
        
        # TODO refactor this function; it is returning stuff for all crews
        constraint_data = define_network_constraint_data(crew_arcs, num_crews, num_fires, num_time_periods)

        base_time = constraint_data.b_in[crew, :, :]
        state_in_arcs = vcat(
            constraint_data.f_in[crew, :, :, :],
            reshape(base_time, (1, size(base_time)...)),
        )[
            :,
            :,
            :,
        ]

        base_time_out = constraint_data.b_out[crew, :, :]
        state_out_arcs = vcat(
            constraint_data.f_out[crew, :, :, :],
            reshape(base_time_out, (1, size(base_time_out)...)),
        )[
            :,
            :,
            :,
        ]

        linking_dual_arc_lookup = Matrix{Vector{Int64}}(undef, num_fires, num_time_periods)
        for g ∈ 1:num_fires
            for t ∈ 1:num_time_periods
                linking_dual_arc_lookup[g, t] = Int64[]
            end
        end

        for i in 1:length(crew_arc_costs)
            if (crew_arcs[i, CM.TO_TYPE] == CM.FIRE_CODE) && (crew_arcs[i, CM.TIME_TO] <= num_time_periods)
                g = crew_arcs[i, CM.LOC_TO]
                t = crew_arcs[i, CM.TIME_TO]
                push!(linking_dual_arc_lookup[g, t], i)
            end
        end

        crew_sp =
            TimeSpaceNetwork(crew_arc_costs, state_in_arcs, state_out_arcs, "crew", crew_arcs, crew_wide_arcs, copy(crew_arc_costs), falses(length(crew_arc_costs)), linking_dual_arc_lookup)
        push!(crew_sps, crew_sp)
    end

    return crew_sps
end



function build_crew_models(
    in_path::String,
    num_fires::Int64,
    num_crews::Int64,
    num_time_periods::Int64,
    travel_speed::Float64
)

    dists_and_times, crew_status = crew_data_from_path(in_path, travel_speed)

    arcs = generate_arcs(
        dists_and_times,
        crew_status,
        num_crews,
        num_fires,
        num_time_periods,
    )

    rest_pen = get_rest_penalties(
        num_crews,
        num_time_periods,
        crew_status.rest_by,
        1e10,
        positive,
    )
    ALPHA = 200
    cost_params = Dict(
        "cost_per_mile" => 1,
        "rest_violation" => rest_pen,
        "fight_fire" => ALPHA,
    )

    crew_sps = TimeSpaceNetwork[]
    for crew in 1:num_crews

        n_arcs = length(arcs[:, 1])
        crew_arcs = arcs[[i for i in 1:n_arcs if arcs[i, CM.CREW_NUMBER] == crew], :]
        crew_wide_arcs = collect(crew_arcs')
        crew_arc_costs = get_static_crew_arc_costs(dists_and_times, crew_arcs, cost_params)
        
        # TODO refactor this function; it is returning stuff for all crews
        constraint_data = define_network_constraint_data(crew_arcs, num_crews, num_fires, num_time_periods)

        base_time = constraint_data.b_in[crew, :, :]
        state_in_arcs = vcat(
            constraint_data.f_in[crew, :, :, :],
            reshape(base_time, (1, size(base_time)...)),
        )[
            :,
            :,
            :,
        ]

        base_time_out = constraint_data.b_out[crew, :, :]
        state_out_arcs = vcat(
            constraint_data.f_out[crew, :, :, :],
            reshape(base_time_out, (1, size(base_time_out)...)),
        )[
            :,
            :,
            :,
        ]

        linking_dual_arc_lookup = Matrix{Vector{Int64}}(undef, num_fires, num_time_periods)
        for g ∈ 1:num_fires
            for t ∈ 1:num_time_periods
                linking_dual_arc_lookup[g, t] = Int64[]
            end
        end

        for i in 1:length(crew_arc_costs)
            if (crew_arcs[i, CM.TO_TYPE] == CM.FIRE_CODE) && (crew_arcs[i, CM.TIME_TO] <= num_time_periods)
                g = crew_arcs[i, CM.LOC_TO]
                t = crew_arcs[i, CM.TIME_TO]
                push!(linking_dual_arc_lookup[g, t], i)
            end
        end

        crew_sp =
            TimeSpaceNetwork(crew_arc_costs, state_in_arcs, state_out_arcs, "crew", crew_arcs, crew_wide_arcs, copy(crew_arc_costs), falses(length(crew_arc_costs)), linking_dual_arc_lookup)
        push!(crew_sps, crew_sp)
    end

    return crew_sps
end

function create_discrete_fire_states(
    params,
    agg_prec,
    passive_states,
    num_time_periods,
)

    if params["model_type"] == "simple_linear"

        # get the no-suppression progression of this fire
        progs = params["progressions"]
        start_perim = params["start_perim"]
        no_supp = [start_perim]
        for i in 1:num_time_periods
            push!(no_supp, no_supp[i] * progs[i])
        end

        # generalize this later
        aggressive_precision = agg_prec
        num_aggressive_states =
            convert(Int, round(start_perim * 2 / aggressive_precision))
        num_passive_states = passive_states

        aggressive_states = LinRange(
            0,
            num_aggressive_states * aggressive_precision,
            num_aggressive_states,
        )
        passive_states =
            exp.(
                LinRange(log(num_aggressive_states * aggressive_precision + 1),
                    maximum(log.(no_supp .+ 1)), num_passive_states + 1)
            )
        passive_states = passive_states[2:num_passive_states+1] .- 1
        all_states = vcat(aggressive_states, passive_states)
        all_states = vcat(all_states, 1e7)

        push!(all_states, start_perim)

    else
        error("Not implemented")
    end

    return all_states
end

function get_state_entrance_cost(state, enter_time, params, num_time_periods)

    if params["model_type"] == "simple_linear"

        if (enter_time == 1) | (enter_time == num_time_periods + 1)
            return params["beta"] * state / 2
        elseif (enter_time > 1) & (enter_time < num_time_periods + 1)
            return params["beta"] * state
        else
            error("Invalid time")
        end
    else
        error("Not implemented")
    end
end

function arcs_from_state_graph(graph)

    visitable = [
        (i, j) for
        i in 1:size(graph)[1], j in 1:size(graph)[2] if isassigned(graph, i, j)
    ]
    edges = []

    for (i, j) in visitable
        push!(
            edges,
            copy(reduce(hcat, [[i, j, j + 1, a[1], a[2]] for a in graph[i, j]])'),
        )
    end

    return convert.(
        Int,
        vcat([size(graph)[1], 0, 1, size(graph)[1], 0]', reduce(vcat, edges)),
    )
end

function get_state_in_arcs(arc_array, n_states, num_time_periods)

    # get the arcs and crew requirements
    n_arcs = length(arc_array[:, 1])

    # get arcs entering (in) and exiting (out) each state
    in_arcs = Array{Vector{Int64}}(undef, n_states, num_time_periods + 1)
    for s in 1:n_states
        state_arcs = [i for i in 1:n_arcs if arc_array[i, FM.STATE_TO] == s]
        for t in 1:num_time_periods+1
            in_arcs[s, t] = [i for i in state_arcs if arc_array[i, FM.TIME_TO] == t]
        end
    end

    return in_arcs
end

function get_state_out_arcs(arc_array, n_states, num_time_periods)

    # get the arcs and crew requirements
    n_arcs = length(arc_array[:, 1])

    # get arcs entering (in) and exiting (out) each state
    out_arcs = Array{Vector{Int64}}(undef, n_states, num_time_periods + 1)
    for s in 1:n_states
        state_arcs = [i for i in 1:n_arcs if arc_array[i, FM.STATE_FROM] == s]
        for t in 1:num_time_periods+1
            out_arcs[s, t] = [i for i in state_arcs if arc_array[i, FM.TIME_FROM] == t]
        end
    end

    return out_arcs
end

function update_fire_stats(curr_stats, curr_time, crew_allocation, params)

    if params["model_type"] == "simple_linear"

        line_per_crew = params["line_per_crew"]
        prog = params["progressions"][curr_time]
        line = line_per_crew * crew_allocation
        next_stats = (curr_stats - line / 2) * prog - line / 2
        next_stats = max(0, next_stats)

    else
        error("Not implemented")
    end

    return next_stats
end

function inverse_update_fire_stats(stats_from, stats_to, time_from, time_to, params)
    """ Returns number of crews needed for given fire transition """

    if params["model_type"] == "simple_linear"

        line_per_crew = params["line_per_crew"]
        prog = params["progressions"][time_from]

        crews = 2 / line_per_crew * (prog * stats_from - stats_to) / (1 + prog)

    else
        error("Not implemented")
    end

    return crews
end

function generate_state_transition_crew_reqs(
    curr_stats,
    curr_time,
    sorted_states,
    params,
    num_crews,
    round_types,
)

    if params["model_type"] == "simple_linear"

        # get all possibly feasible states
        min_state_val =
            update_fire_stats(curr_stats, curr_time, num_crews + 0.5, params)
        max_state_val = update_fire_stats(curr_stats, curr_time, 0, params)

        # get min and max index of the possibly feasible states
        min_state_ix = searchsorted(sorted_states, min_state_val).start
        max_state_ix = searchsorted(sorted_states, max_state_val).start

        # inititalize output 
        output = Dict()
        for round_type in round_types
            output[round_type] = []
        end

        # inititalize minimum crews needed so far, for trimming (explained below)
        min_used = Dict()
        for round_type in round_types
            min_used[round_type] = num_crews + 1
        end

        # for each feasible state
        for state_ix in min_state_ix:max_state_ix
            crews_needed =
                inverse_update_fire_stats(curr_stats, sorted_states[state_ix],
                    curr_time,
                    curr_time + 1, params)

            # for each round type
            for round_type in round_types

                # round the number of crews
                if round_type == "ceiling"
                    crews = max(0, convert(Int, ceil(crews_needed - 0.0001)))
                elseif round_type == "floor"
                    crews = max(0, convert(Int, floor(crews_needed + 0.0001)))
                elseif round_type == "nearest"
                    crews = max(0, convert(Int, round(crews_needed)))
                else
                    error("Round type not implemented")
                end

                # since the states are sorted in increasing level of cost and risk
                # we can trim arcs that are dominated

                # if this is a feasible number of crews and we do not have a dominating arc
                if (crews <= num_crews) & (crews < min_used[round_type])

                    # update the minimum crews used for this round type
                    min_used[round_type] = crews

                    # push the crew requirement for this transition to the corresponding list
                    push!(output[round_type], (state_ix, crews))
                end
            end
        end

        return output


    else
        error("Model type not implemented")
    end

end

function generate_graphs(states, params, num_crews, num_time_periods, round_types)

    # separate out non start states
    non_start_states = states[1:length(states)-1]

    # initializations
    n_states = length(non_start_states)
    crews_needed = Dict()

    # for each round type
    for round_type in round_types

        # init graph as array of vectors (each index is a state*time vertex, each vector holds edges out)
        crews_needed[round_type] =
            Array{Vector{}}(undef, n_states + 1, num_time_periods + 1)

        # initialize time, state indices to check next
        curr_time = 1
        next_to_check = [n_states + 1]

        # while we have not yet reached the end of the horizon
        while curr_time < num_time_periods + 1

            # copy the indices to check and reset next_to_check
            to_check = copy(next_to_check)
            next_to_check = []

            # for each state index feasible to reach at this time period
            for check in to_check

                # if it is not the last (no return) state
                if check != n_states

                    # find the crew requirements for feasible edges
                    edges =
                        generate_state_transition_crew_reqs(states[check], curr_time,
                            non_start_states,
                            params, num_crews, [round_type])[round_type]

                    # otherwise stay at this state
                else
                    edges = [(n_states, 0)]
                end

                # update the crews needed for each edge coming out of this state
                crews_needed[round_type][check, curr_time] = edges

                # append the neighbors to next_to_check
                next_to_check = vcat(
                    next_to_check,
                    [
                        edges[i][1] for i in 1:length(edges)
                        if ~(edges[i][1] in next_to_check)
                    ],
                )
            end

            # add one to the time
            curr_time += 1
        end
    end

    return crews_needed
end

function discretize_fire_model(
    config,
    agg_prec,
    passive_states,
    num_crews,
    num_time_periods,
    round_types,
)
    # make graphs 
    states = create_discrete_fire_states(
        config,
        agg_prec,
        passive_states,
        num_time_periods,
    )

    # get costs to enter each state
    state_costs = zeros(length(states), num_time_periods + 1)
    for i ∈ eachindex(states)
        for j ∈ 1:num_time_periods+1
            state_costs[i, j] =
                get_state_entrance_cost(states[i], j, config, num_time_periods)
        end
    end

    graphs =
        generate_graphs(states, config, num_crews, num_time_periods, round_types)
    arc_arrays = Dict()
    fire_arc_costs_dict = Dict()
    in_arcs = Dict()
    out_arcs = Dict()
    for key in keys(graphs)
        arc_array = arcs_from_state_graph(graphs[key])
        n_arcs = length(arc_array[:, 1])

        # add crew to front (actually, this is deprecated, adding just -1's to keep column order)
        arc_array =
            hcat(convert.(Int, zeros(length(arc_array[:, 1]))) .- 1, arc_array)

        fire_arc_costs_dict[key] =
            [state_costs[arc_array[i, 5], arc_array[i, 4]] for i in 1:n_arcs]
        fire_arc_costs_dict[key] = fire_arc_costs_dict[key] ./ 1e6 # scale for Gurobi numerical tolerance
        arc_arrays[key] = copy(arc_array)

        n_states = size(states)[1]
        in_arcs[key] = get_state_in_arcs(arc_arrays[key], n_states, num_time_periods)
        out_arcs[key] = get_state_out_arcs(arc_arrays[key], n_states, num_time_periods)
    end

    return states, graphs, arc_arrays, fire_arc_costs_dict, state_costs, in_arcs, out_arcs
end

function build_fire_models(
    in_path::String,
    num_fires::Int64,
    num_crews::Int64,
    num_time_periods::Int64,
    line_per_crew::Int64,
)

    # get inital fire perimeters and no-suppression progression parameters
    M = readdlm(in_path * "/sample_growth_patterns.csv", ',')
    start_perims = M[:, 1]
    progressions = M[:, 2:15]

    fire_models = TimeSpaceNetwork[]
    round_type = "ceiling"
    agg_prec = 10
    passive_states = 30
    for fire in 1:num_fires
        model_config = Dict("model_type" => "simple_linear",
            "progressions" => progressions[fire, :],
            "start_perim" => start_perims[fire], "line_per_crew" => line_per_crew,
            "beta" => 100)
        _, _, arc_arrays, arc_costs, _, in_arcs, out_arcs = discretize_fire_model(
            model_config,
            agg_prec,
            passive_states,
            num_crews,
            num_time_periods,
            [round_type],
        )

        # need +1 for the start arcs, tracking times {0, ..., T} but Julia uses 1-indexing
        linking_dual_arc_lookup = Matrix{Vector{Int64}}(undef, num_fires, num_time_periods + 1)
        for g ∈ 1:num_fires
            for t ∈ 1:num_time_periods+1
                linking_dual_arc_lookup[g, t] = Int64[]
            end
        end

        for i ∈ 1:length(arc_costs[round_type])
            t = arc_arrays[round_type][i, FM.TIME_FROM] + 1
            push!(linking_dual_arc_lookup[fire, t], i)
        end

        fire_model = TimeSpaceNetwork(
            arc_costs[round_type],
            in_arcs[round_type],
            out_arcs[round_type],
            "fire",
            arc_arrays[round_type],
            collect(arc_arrays[round_type]'),
            copy(arc_costs[round_type]),
            falses(length(arc_costs[round_type])),
            linking_dual_arc_lookup,
        )
        push!(fire_models, fire_model)
    end
    return fire_models
end

function build_fire_models_from_empirical(
    num_fires::Int64, 
    num_crews::Int64, 
    num_time_periods::Int64,
)

    # initialize fire models
    fire_models = TimeSpaceNetwork[]

    # need +1 for the start arcs, tracking times {0, ..., T} but Julia uses 1-indexing
    linking_dual_arc_lookup = Matrix{Vector{Int64}}(undef, num_fires, num_time_periods + 1)
    for g ∈ 1:num_fires
        for t ∈ 1:num_time_periods+1
            linking_dual_arc_lookup[g, t] = Int64[]
        end
    end

    # read in the selected fires
    fire_folder = "data/empirical_fire_models/raw/arc_arrays"
    selected_fires = CSV.read(fire_folder * "/" * "selected_fires.csv", DataFrame)

    # read in the base-to-fire distances
    base_fire_dists = CSV.read(fire_folder * "/../" * "consolidated_distances.csv", DataFrame)

    # restrict to the fires whose fire_id is in the arc_file column of "selected_fires.csv"
    base_fire_dists = base_fire_dists[findall(in(selected_fires[:, "fire_id"]), base_fire_dists[:, "fire_id"]), :]
    
    # turn the base_fire_dists into a matrix, dropping the columns names and the fire_id column
    base_fire_dists = Matrix(base_fire_dists[:, 2:end])

    # transpose the matrix so that the rows are the crews and the columns are the fires
    base_fire_dists = copy(base_fire_dists')

    # TODO fix fire-distances
    # for now they will all be the same, the mean of the base-fire distances
    fire_dists = zeros(num_fires, num_fires)
    dist = sum(base_fire_dists) / length(base_fire_dists)
    for i in 1:num_fires
        for j in 1:num_fires
            if i != j
                fire_dists[i, j] = dist
            end
        end
    end

    for fire in 1:num_fires

        # read in the arc array
        fname = selected_fires[fire, "arc_file"]
        arc_array = readdlm(fire_folder * "/" * fname, ',')

        # divide the last column by 20 (number of personnel per crew)
        arc_array[:, end] = arc_array[:, end] / 20

        # cast to integer, rounding to nearest
        arc_array = convert(Array{Int64}, round.(arc_array))

        # need to append a dummy column to the left hand side
        arc_array = hcat(convert.(Int, zeros(length(arc_array[:, 1]))) .- 1, arc_array)
        
        # read the arc costs
        fname = selected_fires[fire, "cost_file"]
        arc_costs = readdlm(fire_folder * "/" * fname, ',')

        # we need to append a zero-cost arc for the start
        start_location = arc_array[1, FM.STATE_FROM]
        new_arc = [-1, start_location, 0, 1, start_location, 0]
        arc_array = vcat(new_arc', arc_array)
        arc_costs = vcat(0, arc_costs)

        # we should cull the arrays and costs to only include arcs that are feasible
        # for the given number of crews
        feasible_arcs = [i for i in 1:length(arc_array[:, 1]) if arc_array[i, FM.TIME_FROM] <= num_time_periods]
        arc_array = arc_array[feasible_arcs, :]
        arc_costs = arc_costs[feasible_arcs]
        feasible_arcs = [i for i in 1:length(arc_array[:, 1]) if arc_array[i, FM.CREWS_PRESENT] <= num_crews]
        arc_array = arc_array[feasible_arcs, :]
        arc_costs = arc_costs[feasible_arcs]

        # normalize the costs
        arc_costs = arc_costs ./ 1e4 # scale for Gurobi numerical tolerance

        # we need to rename the states to be 1:s for some s
        all_states = unique(vcat(arc_array[:, FM.STATE_FROM], arc_array[:, FM.STATE_TO]))
        num_states = length(all_states)
        state_dict = Dict()
        for (i, state) in enumerate(all_states)
            state_dict[state] = i
        end
        for i in 1:length(arc_array[:, 1])
            arc_array[i, FM.STATE_FROM] = state_dict[arc_array[i, FM.STATE_FROM]]
            arc_array[i, FM.STATE_TO] = state_dict[arc_array[i, FM.STATE_TO]]
        end

        # for each arc, we need to update the linking_dual_arc_lookup
        for i ∈ 1:length(arc_costs)
            t = arc_array[i, FM.TIME_FROM] + 1
            push!(linking_dual_arc_lookup[fire, t], i)
        end

        # define the state_in_arcs and state_out_arcs
        in_arcs = get_state_in_arcs(arc_array, num_states, num_time_periods)
        out_arcs = get_state_out_arcs(arc_array, num_states, num_time_periods)
   
        fire_model = TimeSpaceNetwork(
            arc_costs,
            in_arcs,
            out_arcs,
            "fire",
            arc_array,
            collect(arc_array'),
            copy(arc_costs),
            falses(length(arc_costs)),
            linking_dual_arc_lookup,
        )
        push!(fire_models, fire_model)
    end

    return fire_models
end