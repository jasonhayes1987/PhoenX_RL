import os
import json
import logging
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
import wandb

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.debug("Initializing wandb_provider.py")

# Check if W&B API key is set
if 'WANDB_API_KEY' not in os.environ:
    raise ValueError("WANDB_API_KEY environment variable is not set. Please set it with your Weights & Biases API key.")

# Initialize W&B API
api = wandb.Api()
logging.debug("W&B API initialized")

# Create FastMCP server with W&B dependency
mcp = FastMCP("WandB Analyzer", dependencies=["wandb"])
logging.debug("FastMCP server instance created")

# Resources

@mcp.resource("wandb://sweeps/{entity}/{project}")
def get_sweeps(entity: str, project: str) -> str:
    logging.debug("Registering resource: get_sweeps with entity='%s', project='%s'", entity, project)
    try:
        sweeps = api.sweeps(f"{entity}/{project}")
        sweep_ids = [sweep.id for sweep in sweeps]
        logging.debug("get_sweeps returning: %s", sweep_ids)
        return json.dumps(sweep_ids)
    except Exception as e:
        logging.error("Error in get_sweeps: %s", str(e))
        raise McpError(f"Error fetching sweeps: {str(e)}")

@mcp.resource("wandb://sweep/{entity}/{project}/{sweep_id}/runs")
def get_runs_in_sweep(entity: str, project: str, sweep_id: str) -> str:
    logging.debug("Registering resource: get_runs_in_sweep with entity='%s', project='%s', sweep_id='%s'", entity, project, sweep_id)
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        run_ids = [run.id for run in runs]
        logging.debug("get_runs_in_sweep returning: %s", run_ids)
        return json.dumps(run_ids)
    except Exception as e:
        logging.error("Error in get_runs_in_sweep: %s", str(e))
        raise McpError(f"Error fetching runs in sweep: {str(e)}")

@mcp.resource("wandb://run/{entity}/{project}/{run_id}/config")
def get_run_config(entity: str, project: str, run_id: str) -> str:
    logging.debug("Registering resource: get_run_config with entity='%s', project='%s', run_id='%s'", entity, project, run_id)
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        logging.debug("get_run_config returning: %s", run.config)
        return json.dumps(run.config)
    except Exception as e:
        logging.error("Error in get_run_config: %s", str(e))
        raise McpError(f"Error fetching run config: {str(e)}")

@mcp.resource("wandb://run/{entity}/{project}/{run_id}/summary")
def get_run_summary(entity: str, project: str, run_id: str) -> str:
    logging.debug("Registering resource: get_run_summary with entity='%s', project='%s', run_id='%s'", entity, project, run_id)
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        logging.debug("get_run_summary returning summary")
        return json.dumps(run.summary._json_dict)
    except Exception as e:
        logging.error("Error in get_run_summary: %s", str(e))
        raise McpError(f"Error fetching run summary: {str(e)}")

@mcp.resource("wandb://sweep/{entity}/{project}/{sweep_id}/config")
def get_sweep_config(entity: str, project: str, sweep_id: str) -> str:
    logging.debug("Registering resource: get_sweep_config with entity='%s', project='%s', sweep_id='%s'", entity, project, sweep_id)
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        logging.debug("get_sweep_config returning sweep config")
        return json.dumps(sweep.config)
    except Exception as e:
        logging.error("Error in get_sweep_config: %s", str(e))
        raise McpError(f"Error fetching sweep config: {str(e)}")

# Tools

@mcp.tool()
def get_best_run_in_sweep(entity: str, project: str, sweep_id: str, metric: str) -> dict:
    logging.debug("Registering tool: get_best_run_in_sweep with entity='%s', project='%s', sweep_id='%s', metric='%s'", entity, project, sweep_id, metric)
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        if not runs:
            raise McpError("No runs found in sweep")
        best_run = max(runs, key=lambda run: run.summary.get(metric, float('-inf')))
        logging.debug("Best run in sweep found: %s", best_run.id)
        return {
            "run_id": best_run.id,
            "config": best_run.config,
            "summary": best_run.summary._json_dict
        }
    except Exception as e:
        logging.error("Error in get_best_run_in_sweep: %s", str(e))
        raise McpError(f"Error finding best run in sweep: {str(e)}")

@mcp.tool()
def get_best_run_in_project(entity: str, project: str, metric: str) -> dict:
    """Returns details of the best run in a given project based on the specified metric."""
    logging.debug("Registering tool: get_best_run_in_project with entity='%s', project='%s', metric='%s'", entity, project, metric)
    try:
        runs = api.runs(f"{entity}/{project}")
        runs_with_metric = [run for run in runs if metric in run.summary]
        if not runs_with_metric:
            raise McpError(f"No runs with metric '{metric}' found in project")
        best_run = max(runs_with_metric, key=lambda run: run.summary[metric])
        logging.debug("Best run in project found: %s", best_run.id)
        return {
            "run_id": best_run.id,
            "config": best_run.config,
            "summary": best_run.summary._json_dict
        }
    except Exception as e:
        logging.error("Error in get_best_run_in_project: %s", str(e))
        raise McpError(f"Error finding best run in project: {str(e)}")

# Run the server
if __name__ == "__main__":
    logging.debug("Starting MCP server with stdio transport")
    # mcp.run(transport='stdio')
    mcp.run(transport='sse')