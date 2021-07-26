from azureml.core.compute import ComputeTarget
from azureml.core import Experiment, Environment, ScriptRunConfig, Workspace

def submit():
    # define workspace
    ws = Workspace.from_config()

    # reference existing compute target
    cluster_name = "first-compute"

    print(f"Using existing cluster - {cluster_name}")
    target = ComputeTarget(workspace=ws, name=cluster_name)
    
    target.wait_for_completion(show_output=True)

    # create an environment using an existing conda file
    env = Environment.from_conda_specification(name="europython-env", file_path="env.yml")

    # create script run configuration
    src = ScriptRunConfig(source_directory=".", script="train.py",
        compute_target=target, environment=env)

    src.run_config.target = target

    # create an experiment
    experiment_name = "Default"
    experiment = Experiment(workspace=ws, name=experiment_name)

    # run experiment
    run = experiment.submit(config=src)
    run.wait_for_completion(show_output=True)

    return True

if __name__ == "__main__":
    submit()

    
