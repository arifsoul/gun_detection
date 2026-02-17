import mlflow
import os
import shutil
import tempfile
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def migrate_full_mlflow():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_uri = f"file:///{os.path.join(project_root, 'mlruns')}"
    db_path = os.path.join(project_root, "mlflow.db")
    dest_uri = f"sqlite:///{db_path}"

    print(f"Migration Source: {source_uri}")
    print(f"Migration Destination: {dest_uri}")

    # Check if DB exists and warn
    if os.path.exists(db_path):
        print("\nWARNING: mlflow.db already exists.")
        print(
            "To avoid duplicates, it is recommended to delete mlflow.db before running this script"
        )
        print("unless you are merging into an existing database.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != "y":
            print("Migration Aborted.")
            return

    # Clients
    client_src = MlflowClient(tracking_uri=source_uri)
    client_dest = MlflowClient(tracking_uri=dest_uri)

    # List Experiments
    experiments = client_src.search_experiments(view_type=ViewType.ALL)
    print(f"Found {len(experiments)} experiments.")

    for exp in experiments:
        print(f"\nProcessing Experiment: {exp.name} (ID: {exp.experiment_id})")

        # Get or Create Experiment in Dest
        try:
            dest_exp_id = client_dest.create_experiment(
                exp.name, artifact_location=exp.artifact_location
            )
            print(
                f"  Created destination experiment '{exp.name}' with ID {dest_exp_id}"
            )
        except Exception:
            dest_exp = client_dest.get_experiment_by_name(exp.name)
            if dest_exp:
                dest_exp_id = dest_exp.experiment_id
                print(
                    f"  Destination experiment '{exp.name}' already exists with ID {dest_exp_id}"
                )
            else:
                print(f"  Failed to create/get experiment '{exp.name}'. Skipping.")
                continue

        # List Runs
        runs = client_src.search_runs(exp.experiment_id)
        print(f"  Found {len(runs)} runs in experiment '{exp.name}'")

        for run in runs:
            run_id = run.info.run_id
            print(f"    Migrating Run {run_id}...")

            # Create Run in Dest
            try:
                # Create a new run in the destination
                dest_run = client_dest.create_run(
                    dest_exp_id, start_time=run.info.start_time, tags=run.data.tags
                )
                dest_run_id = dest_run.info.run_id

                # 1. Log Params
                if run.data.params:
                    # MLflow log_param can take single k,v. For bulk, we'd need log_batch but client.log_param is simple enough for migration loop
                    for key, value in run.data.params.items():
                        client_dest.log_param(dest_run_id, key, value)

                # 2. Log Metrics (Full History)
                if run.data.metrics:
                    for key in run.data.metrics.keys():
                        # Get full history for this metric
                        history = client_src.get_metric_history(run_id, key)
                        for m in history:
                            client_dest.log_metric(
                                dest_run_id, m.key, m.value, m.timestamp, m.step
                            )

                # 3. Log Artifacts
                # We need a temp directory to download artifacts to, then upload them
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        # Download all artifacts from source run
                        local_path = client_src.download_artifacts(
                            run_id, path="", dst_path=tmp_dir
                        )

                        # List what we downloaded to verify (optional, mostly for debugging if needed)
                        # ignored_files = os.listdir(local_path)

                        # Upload to destination run
                        # log_artifacts logs the *contents* of the directory
                        client_dest.log_artifacts(dest_run_id, local_path)
                    except Exception as art_err:
                        print(
                            f"      Warning: Could not migrate artifacts for run {run_id}: {art_err}"
                        )

                # 4. Terminate Run (Status & End Time)
                client_dest.set_terminated(
                    dest_run_id,
                    status=run.info.status,
                    end_time=run.info.end_time,
                )

            except Exception as e:
                print(f"      Failed to migrate run {run_id}: {e}")

    print("\nMigration Complete.")


if __name__ == "__main__":
    migrate_full_mlflow()
