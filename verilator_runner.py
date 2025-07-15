"""
Docker Verilator Runner Module

This module provides functionality to run Docker Compose with Verilator
in a specified target directory with proper error handling and cleanup.
"""

import os
import shutil
import subprocess
import textwrap


def run_docker_compose(*, target_dir=str, strip_lines=False) -> int:
    """
    Run Docker Compose with Verilator in the specified target directory.

    Args:
        target_dir (str): Path to the target directory containing .env file
        strip_lines (bool): Whether to strip first and last lines from output

    Returns:
        int:
            Return code from the Docker process,
            0 for success, non-zero for failure.
    """
    obj_dir = target_dir + "obj_dir"

    if os.path.isdir(target_dir):
        env_file_path = os.path.join(target_dir, ".env")

        if not os.path.exists(env_file_path):
            print(f"Error: Environment file not found at {env_file_path}")
            return 1
        else:
            print("Verilator Simulation Output:")
            print("=" * 80)

            docker_output = []
            process = subprocess.Popen(
                [
                    "docker-compose",
                    "--env-file",
                    env_file_path,
                    "run",
                    "--rm",
                    "verilator",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Get all lines from process output
            all_lines = list(process.stdout)

            # Apply stripping if enabled
            if (
                strip_lines and len(all_lines) > 18
            ):  # Need at least 18 lines to strip 14 from start and 4 from end
                lines_to_process = all_lines[14:-4]
            else:
                lines_to_process = all_lines

            # Process the lines
            for line in lines_to_process:
                clean_line = line.rstrip()
                docker_output.append(clean_line)
                if len(clean_line) > 80:
                    wrapped_lines = textwrap.fill(clean_line, width=80)
                    print(wrapped_lines)
                else:
                    print(clean_line)

            process.wait()
            print("=" * 80)
            print(f"Process finished with return code: {process.returncode}")

            # Check for specific Docker connection issues
            if process.returncode != 0:
                output_text = " ".join(docker_output)
                if (
                    "dockerDesktopLinuxEngine" in output_text
                    or "cannot find the file specified" in output_text
                ):
                    print("\n🐳 Docker Connection Error Detected:")
                    print("- Please start Docker Desktop")
                    print("- Wait for Docker to fully initialize")
                    print(
                        "- Try running 'docker ps' "
                        "to verify Docker is working"
                    )
                else:
                    print("\nDocker command failed. Common issues:")
                    print("- Docker Desktop is not running")
                    print("- Docker daemon is not accessible")
                    print("- Docker service needs to be started")

            # Cleanup
            if os.path.isdir(obj_dir):
                print(f"Removing {obj_dir} directory...")
                shutil.rmtree(obj_dir)
                print(f"{obj_dir} removed successfully.")
            else:
                print(f"{obj_dir} directory not found, skipping cleanup.")

            return process.returncode
    else:
        print(f"Directory not found: {target_dir}")
        return 1


if __name__ == "__main__":
    pass
