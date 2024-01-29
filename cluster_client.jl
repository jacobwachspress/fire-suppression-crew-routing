include("environment.jl")
include("cluster_test_helper.jl")
include("DCG.jl")

using JSON
const GRB_ENV = Gurobi.Env()

function run_experiment(date, json_name, write_output)
    
    # get path to input parameters
    json_path = "data/experiment_inputs/" * date * "/" * json_name * ".json"
    
    # read in parameters
    external_params = JSON.parse(open(json_path))

    # create full dict of parameters
    dcg_params = default_params()
    for key in keys(external_params["dcg"])
        dcg_params[key] = external_params["dcg"][key]
    end

    # update global variables
    global NUM_FIRES = dcg_params["num_fires"]
    global NUM_CREWS = dcg_params["num_crews"]
    global LINE_PER_CREW = dcg_params["line_per_crew"]

    # set perturbations in int-aware-plan generating phase
    capacity_perturbations = [-2, -1, 0, 1, 2] .* (NUM_CREWS / 20)

    # process all the relevant location, fire data
    preprocessed_data = preprocess(dcg_params["in_path"])

    # run DCG at root node, saving dual warm start for future iterations
    t = @elapsed d, cg_data = single_DCG_node(dcg_params, deepcopy(preprocessed_data))
    @assert dcg_params["dual_warm_start"] != "calculate"
    
    # update int-aware capacities from weighted average primal solution
    dcg_params["int_aware_capacities"] = d["allotments"]["master_problem_reconstructed"]
    
    # generate int-aware plans
    iterations, timings, cg_data = generate_new_plans(dcg_params, preprocessed_data, cg_data, capacity_perturbations, -1, 0)
    
    # restore integrality
    form_time, sol_time, pb, form_time_2, sol_time_2, pb_2 = restore_integrality(cg_data, 3600);
    
    # write output to JSON
    if write_output
        
        out_dir = "data/experiment_outputs/" * date * "/"
        if (~isdir(out_dir))
            mkpath(out_dir)
        end
        
        outputs = Dict{String, Any}()
        delete!(d, "mp")
        outputs["initial_DCG"] = d
        outputs["generate_additional_plans"] = Dict{String, Any}()
        outputs["generate_additional_plans"]["iterations"] = iterations
        outputs["generate_additional_plans"]["timings"] = timings
        outputs["cg_data"] = Dict{String, Any}()
        outputs["cg_data"]["routes"] = cg_data.routes.routes_per_crew
        outputs["cg_data"]["plans"] = cg_data.suppression_plans.plans_per_fire
        outputs["restore_integrality"] = Dict{String, Any}()
        outputs["restore_integrality"]["formulation_time"] = form_time
        outputs["restore_integrality"]["solve_time"] = sol_time
        outputs["restore_integrality"]["pb_objective"] = objective_value(pb["m"])
        outputs["restore_integrality"]["pb_objective_bound"] = objective_bound(pb["m"])
        outputs["restore_integrality"]["formulation_time_new"] = form_time_2
        outputs["restore_integrality"]["solve_time_new"] = sol_time_2
        outputs["restore_integrality"]["pb_objective_new"] = objective_value(pb_2["m"])
        outputs["restore_integrality"]["pb_objective_bound_new"] = objective_bound(pb_2["m"])
        
        
        open(out_dir * json_name * ".json", "w") do f
            JSON.print(f, outputs, 4)
        end
  
    end
    
    return 1  
end
    

date = ARGS[1]
number = ARGS[2]
run_experiment(string(date), "precompile", false)
run_experiment(string(date), string(number), true)

