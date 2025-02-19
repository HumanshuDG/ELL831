import csv
import itertools
import logging
import multiprocessing as mp
import os
import random
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np


@dataclass
class SimulationResult:
    avg_power: float
    t_rise_50: float
    delay_ps: float
    pdp: float

    def objective_function(
        self,
        weights: Dict[str, float],
        params: Dict[str, float],
        reg_weight: float = 0.0,
    ) -> float:
        """
        Calculate weighted sum of metrics with regularization.

        Args:
            weights: Dictionary mapping metric names to their weights
            params: Current parameter values for regularization
            reg_weight: Weight for L2 regularization

        Returns:
            Weighted sum of metrics with regularization (negative because we're minimizing)
        """
        weighted_sum = (
            weights.get("avg_power", 1.0) * self.avg_power
            + weights.get("t_rise_50", 1.0) * self.t_rise_50
            + weights.get("delay_ps", 1.0) * self.delay_ps
            + weights.get("pdp", 1.0) * self.pdp
        )

        # Add L2 regularization term
        reg_term = reg_weight * sum(max(p, -p) for p in params.values())

        return -(weighted_sum + reg_term)  # Negative because we're maximizing


def _compute_partial_derivative(
    param_info,
    current_params,
    epsilon,
    optimizer,
    working_dir,
    weights,
    current_objective,
):
    """Compute partial derivative using only forward difference to reduce simulation count"""
    param_name, current_value = param_info

    # Create parameter set for +epsilon only
    params_plus = current_params.copy()
    print(f"current_value: {current_value}, epsilon: {epsilon}")
    params_plus[param_name] = current_value + epsilon
    print(f"params_plus: {params_plus}")

    # Create modified NET file
    net_plus = working_dir / f"{param_name}_plus.net"
    optimizer._create_modified_net(params_plus, net_plus)

    # Return the net file path and parameter info for batch processing
    return (net_plus, param_name, current_objective)


class OptimizationResult(NamedTuple):
    params: Dict[str, float]
    objective: float
    metrics: SimulationResult


def setup_logger(log_file: str = "optimization.log") -> logging.Logger:
    """Configure logging for the optimizer"""
    logger = logging.getLogger("LTSpiceOptimizer")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class LTSpiceOptimizer:
    def __init__(
        self,
        net_file: str,
        ltspice_path: str = "~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
        epsilon: float = 0.01,
        initial_learning_rate: float = 0.1,
        max_iterations: int = 100,
        weights: Optional[Dict[str, float]] = None,
        csv_log_file: str = "optimization_results.csv",
        regularization_weight: float = 1e-4,
    ):
        # Convert paths to Path objects and expand user directory
        self.net_file = Path(os.path.expanduser(net_file))
        self.ltspice_path = Path(os.path.expanduser(ltspice_path))
        self.epsilon = epsilon
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.weights = weights or {
            "avg_power": 0.1,
            "t_rise_50": 10.0,
            "delay_ps": 10.0,
            "pdp": 1.0,
        }
        self.csv_log_file = csv_log_file
        self.params = self._extract_params()
        self.regularization_weight = regularization_weight
        self.logger = setup_logger()

        # Initialize CSV log file with header
        with open(self.csv_log_file, "w", newline="") as f:
            fieldnames = (
                ["start_id", "iteration", "objective_value"]
                + list(self.params.keys())
                + ["avg_power", "t_rise_50", "delay_ps", "pdp"]
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _extract_params(self) -> Dict[str, float]:
        """Extract length parameters from the .net file"""
        params = {}
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(self.net_file, "r", encoding=encoding) as f:
                    for line in f:
                        if line.startswith(".param"):
                            match = re.match(
                                r"\.param\s+(\w+)\s+([\d.]+)([munpf]?)", line
                            )
                            if match:
                                name, value, unit = match.groups()
                                # Only include length parameters
                                if name.startswith("w"):  # Length parameters only
                                    value_float = float(value)
                                    params[name] = value_float
                    return params
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not read {self.net_file} with any supported encoding")

    def _create_modified_net(self, params: Dict[str, float], output_path: Path) -> None:
        """Create a modified .net file with new parameter values"""
        # Try different encodings for reading
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        content = None

        for encoding in encodings:
            try:
                with open(self.net_file, "r", encoding=encoding) as fin:
                    content = fin.read()
                    break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(
                f"Could not read {self.net_file} with any supported encoding"
            )

        # Process the file line by line
        output_lines = []
        for line in content.splitlines():
            if line.startswith(".param"):
                match = re.match(r"\.param\s+(\w+)\s+([\d.]+)([munpf]?)", line)
                if match and match.group(1) in params:
                    param_name = match.group(1)
                    # Ensure length parameters are positive
                    param_value = params[param_name]
                    if param_name.startswith("w"):  # Length parameter
                        param_value = max(1.0, param_value)  # Minimum 1nm length

                    # Keep the same unit suffix if it exists
                    unit_suffix = match.group(3) or ""
                    line = f".param {param_name} {param_value}{unit_suffix}"
            output_lines.append(line)

        # Write with UTF-8 encoding
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write("\n".join(output_lines))

    def _windows_to_wine_path(self, linux_path: Path) -> str:
        """Convert a Linux path to a Wine Windows path format"""
        linux_path = linux_path.resolve()  # Get absolute path
        wine_prefix = os.path.expanduser("~/.wine")

        try:
            # Get relative path from wine prefix
            rel_path = os.path.relpath(linux_path, wine_prefix)
            self.logger.debug(f"Converting path: {linux_path} -> {rel_path}")

            if rel_path.startswith("drive_c/"):
                # Remove drive_c/ and replace with C:\
                win_path = "C:\\" + rel_path[8:].replace("/", "\\")
            else:
                # For other paths in the Wine prefix
                win_path = "Z:\\" + str(linux_path).replace("/", "\\")

            self.logger.debug(f"Converted to Windows path: {win_path}")
            return win_path

        except ValueError as e:
            self.logger.error(f"Error converting path {linux_path}: {e}")
            # Fallback to Z: drive
            win_path = "Z:\\" + str(linux_path).replace("/", "\\")
            self.logger.debug(f"Fallback Windows path: {win_path}")
            return win_path

    def _run_simulation(self, net_file: Path) -> SimulationResult:
        """Run LTspice simulation through Wine and extract results"""
        # Convert to Wine path format
        wine_path = self._windows_to_wine_path(net_file)

        # Run LTspice in batch mode with the netlist
        cmd = ["wine", str(self.ltspice_path), "-Run", "-wine", "-b", wine_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(self.net_file),
        )

        if result.returncode != 0:
            self.logger.error(f"LTspice simulation failed: {result.stderr}")
            raise RuntimeError("LTspice simulation failed")

        # Parse log file
        log_file = net_file.with_suffix(".log")
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        try:
            with open(log_file, "r") as f:
                content = f.read()
                metrics = self._parse_metrics(content)
                return SimulationResult(*metrics)
        except Exception as e:
            self.logger.error(f"Error processing {log_file}: {str(e)}")
            raise

    def _parse_metrics(self, content: str) -> Tuple[float, float, float, float]:
        """Parse metrics from log file content"""
        avg_power = t_rise_50 = delay_ps = pdp = 0.0

        # Parse each metric
        if "avg_power:" in content:
            match = re.search(r"AVG\(-v\(n001\)\*i\(v1\)\)=([\d.e+-]+)", content)
            if match:
                avg_power = float(match.group(1))

        if "t_rise_50=" in content:
            match = re.search(r"t_rise_50=([\d.e+-]+)", content)
            if match:
                t_rise_50 = float(match.group(1))

        if "delay_ps:" in content:
            match = re.search(r"\(t_rise_50\)=([\d.e+-]+)", content)
            if match:
                delay_ps = float(match.group(1))

        if "pdp:" in content:
            match = re.search(r"\(avg_power\*delay_ps\)=([\d.e+-]+)", content)
            if match:
                pdp = float(match.group(1))

        # Verify all metrics were found
        if any(x == 0.0 for x in [avg_power, t_rise_50, delay_ps, pdp]):
            raise ValueError("Failed to parse all simulation metrics")

        return avg_power, t_rise_50, delay_ps, pdp

    def _run_batch_simulation(self, net_files: List[Path]) -> List[SimulationResult]:
        """Run multiple LTspice simulations in batch mode"""
        # Convert all paths to Wine format
        wine_paths = [self._windows_to_wine_path(net) for net in net_files]

        # Log the command being executed
        cmd = ["wine", str(self.ltspice_path), "-Run", "-wine", "-b"] + wine_paths
        cmd_str = " ".join(cmd)
        self.logger.info(f"Executing LTSpice command: {cmd_str}")

        # Verify input files exist
        for net_file in net_files:
            if net_file.exists():
                self.logger.info(f"Found netlist file: {net_file}")
            else:
                self.logger.error(f"Netlist file not found: {net_file}")

        # Run LTSpice
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(
                self.net_file
            ),  # Run in the same directory as original netlist
        )

        # Log the complete output
        self.logger.info("LTSpice stdout:")
        self.logger.info(result.stdout)
        self.logger.info("LTSpice stderr:")
        self.logger.info(result.stderr)

        if result.returncode != 0:
            self.logger.error(
                f"LTspice batch simulation failed with return code {result.returncode}"
            )
            self.logger.error(f"STDERR: {result.stderr}")

        # Check for log files
        for net_file in net_files:
            log_file = net_file.with_suffix(".log")
            if log_file.exists():
                self.logger.info(f"Log file created: {log_file}")
                self.logger.info(f"Log file size: {log_file.stat().st_size} bytes")
                # Print first few lines of log file for debugging
                try:
                    with open(log_file, "r") as f:
                        first_lines = "".join(f.readlines()[:10])
                        self.logger.info(
                            f"First few lines of {log_file}:\n{first_lines}"
                        )
                except Exception as e:
                    self.logger.error(f"Error reading log file {log_file}: {e}")
            else:
                self.logger.error(f"Log file not created: {log_file}")
                # List directory contents
                dir_contents = list(log_file.parent.glob("*"))
                self.logger.info(f"Directory contents: {dir_contents}")

        # Parse results from all log files
        results = []
        for net_file in net_files:
            log_file = net_file.with_suffix(".log")
            self.logger.info(f"Processing log file: {log_file}")

            # Retry a few times if the log file isn't available immediately
            max_retries = 3
            for retry in range(max_retries):
                try:
                    with open(log_file, "r") as f:
                        avg_power = t_rise_50 = delay_ps = pdp = 0.0
                        for line in f:
                            if "avg_power:" in line:
                                match = re.search(
                                    r"AVG\(-v\(n001\)\*i\(v1\)\)=([\d.e+-]+)", line
                                )
                                if match:
                                    avg_power = float(match.group(1))
                            elif "t_rise_50=" in line:
                                match = re.search(r"t_rise_50=([\d.e+-]+)", line)
                                if match:
                                    t_rise_50 = float(match.group(1))
                            elif "delay_ps:" in line:
                                match = re.search(r"\(t_rise_50\)=([\d.e+-]+)", line)
                                if match:
                                    delay_ps = float(match.group(1))
                            elif "pdp:" in line:
                                match = re.search(
                                    r"\(avg_power\*delay_ps\)=([\d.e+-]+)", line
                                )
                                if match:
                                    pdp = float(match.group(1))

                        if all(x != 0.0 for x in [avg_power, t_rise_50, delay_ps, pdp]):
                            results.append(
                                SimulationResult(avg_power, t_rise_50, delay_ps, pdp)
                            )
                            break
                        else:
                            if retry == max_retries - 1:
                                self.logger.error(
                                    f"Failed to parse metrics from {log_file} after {max_retries} attempts"
                                )
                                results.append(None)
                            else:
                                time.sleep(1)  # Wait before retrying

                except FileNotFoundError:
                    if retry == max_retries - 1:
                        self.logger.error(
                            f"Log file not found after {max_retries} attempts: {log_file}"
                        )
                        results.append(None)
                    else:
                        time.sleep(1)  # Wait before retrying
                except Exception as e:
                    self.logger.error(f"Error processing {log_file}: {str(e)}")
                    results.append(None)
                    break

        return results

    def _run_parallel_simulation(self, param_info_tuple) -> Tuple[str, float]:
        """Run a single simulation in parallel (for gradient computation)"""
        net_file, param_name, optimizer = param_info_tuple
        result = optimizer._run_simulation(net_file)
        objective = result.objective_function(
            optimizer.weights, optimizer.params, optimizer.regularization_weight
        )
        return param_name, objective

    def _compute_gradient(
        self,
        current_params: Dict[str, float],
        working_dir: Path,
        current_objective: float,
    ) -> Dict[str, float]:
        """Compute gradient using sequential simulations"""
        gradient = {}

        for param_name, current_value in current_params.items():
            # Create parameter set for forward difference
            params_plus = current_params.copy()
            new_value = current_value + self.epsilon

            # Ensure length parameters stay positive
            if param_name.startswith("w"):
                new_value = max(1.0, new_value)

            params_plus[param_name] = new_value

            # Create and simulate modified netlist
            net_plus = working_dir / f"{param_name}_plus.net"
            self._create_modified_net(params_plus, net_plus)
            result = self._run_simulation(net_plus)

            # Calculate gradient using forward difference with regularization
            objective_plus = result.objective_function(
                self.weights, params_plus, self.regularization_weight
            )
            partial = (objective_plus - current_objective) / self.epsilon
            gradient[param_name] = partial

        return gradient

    @staticmethod
    def _run_single_optimization(optimizer_params) -> Optional[OptimizationResult]:
        """Run a single optimization with given parameters (for parallel multi-start)"""
        optimizer, initial_params, start_id = optimizer_params
        try:
            return optimizer.optimize_single(initial_params, start_id)
        except Exception as e:
            optimizer.logger.error(f"Optimization failed: {str(e)}")
            return None

    def optimize_multi(self, n_starts: int = 10) -> OptimizationResult:
        """Run multiple optimizations in parallel with different starting points"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting optimization with {n_starts} parallel runs")
        self.logger.info(f"{'='*50}\n")

        # Generate random starting points with start IDs
        initial_params_list = [
            (self, self._generate_random_params(), i + 1) for i in range(n_starts)
        ]

        # Run optimizations in parallel
        max_workers = min(n_starts, mp.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(LTSpiceOptimizer._run_single_optimization, params)
                for params in initial_params_list
            ]

            # Collect results as they complete
            results = []
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        self.logger.info(
                            f"\nRun {i}/{n_starts} completed:"
                            f"\n  Objective: {result.objective:.2e}"
                            f"\n  Power: {result.metrics.avg_power:.2e} W"
                            f"\n  Delay: {result.metrics.delay_ps:.2e} s"
                            f"\n  PDP: {result.metrics.pdp:.2e} J"
                        )
                except Exception as e:
                    self.logger.error(f"Run {i}/{n_starts} failed: {str(e)}")

        if not results:
            raise ValueError("All optimization runs failed")

        # Find best result
        best_result = max(results, key=lambda x: x.objective)

        self.logger.info(f"\n{'='*50}")
        self.logger.info("Optimization complete!")
        self.logger.info(f"{'='*50}")
        self.logger.info("\nBest result:")
        self.logger.info(f"  Objective: {best_result.objective:.2e}")
        self.logger.info(f"  Power: {best_result.metrics.avg_power:.2e} W")
        self.logger.info(f"  Delay: {best_result.metrics.delay_ps:.2e} s")
        self.logger.info(f"  PDP: {best_result.metrics.pdp:.2e} J")
        self.logger.info("\nOptimized parameters:")
        for name, value in best_result.params.items():
            self.logger.info(f"  {name}: {value:.2f}")

        # Save final result
        final_net = self.net_file.parent / f"{self.net_file.stem}_optimized.net"
        self._create_modified_net(best_result.params, final_net)
        self.logger.info(f"\nOptimized circuit saved to: {final_net}\n")

        return best_result

    def _log_to_csv(
        self,
        iteration: int,
        params: Dict[str, float],
        result: SimulationResult,
        objective_value: float,
        start_id: int = 0,
    ) -> None:
        """Log the current iteration results to CSV"""
        row_data = {
            "start_id": start_id,
            "iteration": iteration,
            "objective_value": objective_value,
            "avg_power": result.avg_power,
            "t_rise_50": result.t_rise_50,
            "delay_ps": result.delay_ps,
            "pdp": result.pdp,
        }
        row_data.update(params)

        with open(self.csv_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            writer.writerow(row_data)

    def _generate_random_params(self) -> Dict[str, float]:
        """Generate random initial parameters within reasonable bounds"""
        random_params = {}
        for param_name, base_value in self.params.items():
            # Random value between 0.1x and 10x the original value
            random_params[param_name] = base_value * random.uniform(0.8, 1.2)
        return random_params

    def _adaptive_learning_rate(self, iteration: int, prev_obj: float, curr_obj: float):
        """Adjust learning rate based on optimization progress"""
        if curr_obj > prev_obj:
            self.learning_rate *= 1.1  # Increase if improving
        else:
            self.learning_rate *= 0.5  # Decrease if getting worse

        # Bounds checking
        self.learning_rate = max(
            self.initial_learning_rate * 0.001,
            min(self.initial_learning_rate * 10, self.learning_rate),
        )

    def _calculate_area_penalty(self, params: Dict[str, float]) -> float:
        return 0

    def _grid_search(self, n_points: int = 10) -> None:
        """Perform grid search over parameter space and log results"""
        # Create grid points for each parameter
        param_grids = {}
        for param_name, base_value in self.params.items():
            # Create grid from 0.5x to 1.5x of base value
            param_grids[param_name] = np.linspace(
                base_value * 0.5, base_value * 1.5, n_points
            )

        # Create working directory
        working_dir = Path(self.net_file.parent) / "grid_search_temp"
        working_dir.mkdir(exist_ok=True)

        # Create CSV file for grid search results
        grid_csv = "grid_search_results.csv"
        with open(grid_csv, "w", newline="") as f:
            fieldnames = ["point_type"] + list(self.params.keys()) + ["objective_value"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        try:
            # Evaluate points in grid
            for i, param_values in enumerate(itertools.product(*param_grids.values())):
                params = dict(zip(self.params.keys(), param_values))

                # Create and simulate netlist
                net_file = working_dir / f"grid_point_{i}.net"
                self._create_modified_net(params, net_file)
                result = self._run_simulation(net_file)

                # Calculate objective
                objective = result.objective_function(
                    self.weights, params, self.regularization_weight
                )

                # Log result
                row_data = {"point_type": "grid"}
                row_data.update(params)
                row_data["objective_value"] = objective

                with open(grid_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row_data)

        finally:
            import shutil

            shutil.rmtree(working_dir)

    def optimize_single(
        self, initial_params: Dict[str, float], start_id: int = 0
    ) -> OptimizationResult:
        """Run single optimization with given initial parameters and track path"""
        # Create working directory and copy original netlist
        working_dir = Path(self.net_file.parent) / f"optimization_temp_{os.getpid()}"
        working_dir.mkdir(exist_ok=True)

        # Create working copy of original netlist
        working_netlist = working_dir / self.net_file.name
        import shutil

        shutil.copy2(self.net_file, working_netlist)

        # Store original netlist path and temporarily set it to working copy
        original_netlist = self.net_file
        self.net_file = working_netlist

        try:
            # Ensure initial parameters have valid lengths
            validated_params = initial_params.copy()
            for param_name, value in validated_params.items():
                if param_name.startswith("w"):  # Length parameter
                    validated_params[param_name] = max(1.0, value)  # Minimum 1nm length

            current_params = validated_params
            best_params = current_params.copy()
            best_objective = float("-inf")
            best_result = None
            prev_objective = float("-inf")
            no_improvement_count = 0
            last_best_objective = float("-inf")

            # Log optimization path
            with open("grid_search_results.csv", "a", newline="") as f:
                fieldnames = (
                    ["point_type"] + list(self.params.keys()) + ["objective_value"]
                )
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                for iteration in range(self.max_iterations):
                    try:
                        current_net = working_dir / f"iter_{iteration}.net"
                        self._create_modified_net(current_params, current_net)
                        result = self._run_simulation(current_net)

                        # Calculate objective with regularization
                        objective_value = result.objective_function(
                            self.weights, current_params, self.regularization_weight
                        )

                        # Remove separate regularization calculation since it's now part of objective
                        area_penalty = self._calculate_area_penalty(current_params)
                        objective_value -= area_penalty

                        # Update best parameters if improved
                        if objective_value > best_objective:
                            best_objective = objective_value
                            best_params = current_params.copy()
                            best_result = result
                            no_improvement_count = 0
                            last_best_objective = objective_value
                        else:
                            no_improvement_count += 1
                        # Early stopping for single optimization
                        if (
                            no_improvement_count >= 5
                        ):  # Stop after 5 iterations without improvement
                            self.logger.info(
                                f"Stopping optimization at iteration {iteration + 1} due to no improvement"
                            )
                            break
                        # Reduce logging - only log every 5 iterations or on improvement
                        if iteration % 5 == 0 or objective_value > last_best_objective:
                            self.logger.info(
                                f"Iteration {iteration + 1}: "
                                f"Objective: {objective_value:.2e}, "
                                f"Learning Rate: {self.learning_rate:.2e}"
                            )
                        self._log_to_csv(
                            iteration + 1,
                            current_params,
                            result,
                            objective_value,
                            start_id,
                        )
                        # Adaptive learning rate
                        self._adaptive_learning_rate(
                            iteration, prev_objective, objective_value
                        )
                        prev_objective = objective_value
                        gradient = self._compute_gradient(
                            current_params, working_dir, objective_value
                        )
                        # Update parameters
                        for param_name, grad_value in gradient.items():
                            current_params[param_name] += (
                                self.learning_rate * grad_value
                            )

                        # Log the point in optimization path
                        row_data = {"point_type": f"path_{start_id}"}
                        row_data.update(current_params)
                        row_data["objective_value"] = objective_value
                        writer.writerow(row_data)

                    except Exception as e:
                        self.logger.error(f"Error in iteration {iteration}: {str(e)}")
                        if best_result is not None:
                            return OptimizationResult(
                                best_params, best_objective, best_result
                            )
                        raise

        finally:
            # Restore original netlist path
            self.net_file = original_netlist
            # Clean up working directory
            try:
                shutil.rmtree(working_dir)
            except Exception as e:
                self.logger.error(f"Error cleaning up working directory: {e}")

        if best_result is None:
            raise ValueError("Optimization failed to find any valid results")

        return OptimizationResult(best_params, best_objective, best_result)


if __name__ == "__main__":
    optimizer = LTSpiceOptimizer(
        net_file="~/.wine/drive_c/my_ltspice_files/StrongArmLatch.net",
        ltspice_path="~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
        epsilon=5,
        initial_learning_rate=0.02,
        max_iterations=15,
        weights={
            "avg_power": 1.0e5,
            "t_rise_50": 1.0e11,
            "delay_ps": 1.0e11,
            "pdp": 1.0e16,
        },
        regularization_weight=2,
        csv_log_file="optimization_results_vis.csv",
    )

    # Perform grid search first
    optimizer._grid_search(n_points=10)

    # Then run optimization
    try:
        result = optimizer.optimize_multi(n_starts=8)
        print("\nOptimization complete!")
        print(f"Best objective value: {result.objective:.2e}")
        print("Best parameters:", result.params)
        print("\nFinal metrics:")
        print(f"Average power: {result.metrics.avg_power:.2e}")
        print(f"Rise time: {result.metrics.t_rise_50:.2e}")
        print(f"Delay: {result.metrics.delay_ps:.2e}")
        print(f"PDP: {result.metrics.pdp:.2e}")
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
