import os
import re
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from verilator_runner import run_docker_compose

load_dotenv()

LLM_MODEL = "groq:llama3-70b-8192"

LLM_INSTRUCTIONS = """
You are an expert SystemVerilog code generator.
Generate clean, well-structured SystemVerilog code based on user requirements.

**Coding Standards**:
  - Max 80 chars per line.
  - Use 2-space indent.
  - Include clear, relevant comments.
  - Use meaningful signal names.
  - Follow SystemVerilog best practices.

**Requirements**:
  1. **Design Module**:
     - Create a complete, compilable SystemVerilog module.
  2. **Testbench Module**:
     - Instantiate the design module.
     - Provide stimulus (e.g., clock, reset, inputs).
     - Include basic verification (e.g., output checks, result display).
     - Append '_tb' to module name for consistency.
     - Include VCD file generation with:
       - `$dumpfile("<module_name>_tb.vcd")` to save simulation data.
       - `$dumpvars(0, <module_name>_tb)` to dump all testbench variables.
    - Use a meaningful instance name for all instances.
    - Include a $finish statement to properly end the simulation.

**Testbench Message Requirements**:
  - Include informative $display statements throughout the testbench:
    - Print a header message at the start:
    `$display("=== <Module Name> Testbench Started ===");`
    - Print test phase messages:
    `$display("Testing <specific_functionality>...");`
    - Print input values being applied:
    `$display("Applying inputs: <input_description>");`
    - Print expected vs actual results:
    `$display("Expected: %d, Got: %d", expected, actual);`
    - Print pass/fail status for each test:
    `$display("Test <test_name>: %s", pass ? "PASSED" : "FAILED");`
    - Print a summary at the end: `$display("=== Testbench Completed ===");`
  - Use $time in messages where relevant:
  `$display("Time %0t: <message>", $time);`
  - Include error messages for failed assertions or unexpected behavior
  - Make messages clear and descriptive to help with debugging

**Output Format**:
  - Provide exactly two code blocks in order:
    - ```systemverilog design
      <Design module code here>
      ```
    - ```systemverilog testbench
      <Testbench module code here>
      ```
  - Ensure both blocks are present and correctly labeled.
  - Do not include other code blocks or markers.
"""


class AgentState(TypedDict):
    """
    State for SystemVerilog code generation agent.

    Attributes:
        user_request: User request for SystemVerilog code.
        generated_code: Generated SystemVerilog design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        messages: List of conversation messages.
        error: Any error messages during code generation.
        output_dir: Directory for saving files.
        module_dir: Full path to the module-specific directory.
        saved_files: Dictionary tracking saved file paths.
    """

    user_request: str
    generated_code: str
    testbench_code: str
    env_content: str
    messages: Annotated[list, add_messages]
    error: str
    output_dir: str
    module_dir: str
    saved_files: Dict[str, str]


@tool
def load_saved_code_tool(module_dir: str) -> Dict[str, Any]:
    """
    Load previously saved SystemVerilog code from files.

    Args:
        module_dir: Directory containing the saved SystemVerilog files.

    Returns:
        Dict:
            Contains loaded design_code, testbench_code, env_content,
            and any error.
    """
    try:
        result = {
            "design_code": "",
            "testbench_code": "",
            "env_content": "",
            "module_name": "",
            "error": "",
        }

        if not os.path.exists(module_dir):
            return {
                **result,
                "error": f"Module directory does not exist: {module_dir}",
            }

        # Find .env file to get module information
        env_path = os.path.join(module_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                env_content = f.read().strip()
                result["env_content"] = env_content

                # Extract module name from env content
                for line in env_content.split("\n"):
                    if line.startswith("DESIGN_FILE="):
                        design_filename = line.split("=")[1]
                        result["module_name"] = design_filename.replace(
                            ".sv", ""
                        )
                        break

        # If no module name found from .env, try to find .sv files
        if not result["module_name"]:
            sv_files = [
                f
                for f in os.listdir(module_dir)
                if f.endswith(".sv") and not f.endswith("_tb.sv")
            ]
            if sv_files:
                result["module_name"] = sv_files[0].replace(".sv", "")

        if not result["module_name"]:
            return {
                **result,
                "error": "Could not determine module name from saved files",
            }

        # Load design code
        design_path = os.path.join(module_dir, f"{result['module_name']}.sv")
        if os.path.exists(design_path):
            with open(design_path, "r") as f:
                result["design_code"] = f.read().strip()
        else:
            return {
                **result,
                "error": f"Design file not found: {design_path}",
            }

        # Load testbench code
        testbench_path = os.path.join(
            module_dir, f"{result['module_name']}_tb.sv"
        )
        if os.path.exists(testbench_path):
            with open(testbench_path, "r") as f:
                result["testbench_code"] = f.read().strip()
        else:
            result["error"] = f"Testbench file not found: {testbench_path}"

        return result

    except Exception as error:
        return {
            "design_code": "",
            "testbench_code": "",
            "env_content": "",
            "module_name": "",
            "error": f"Error loading saved code: {str(error)}",
        }


@tool
def generate_env_file_tool(
    generated_code: str, output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Generate .env file content for the SystemVerilog project.

    Args:
        generated_code:
            Generated design code from which to extract module name.
        output_dir: Base output directory where files are saved.

    Returns:
        Dict: Contains env_content and any error message.
    """
    try:
        # Extract module name from design code
        module_match = re.search(r"module\s+(\w+)", generated_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )

        # Create relative path to the module directory
        module_relative_path = f"./{output_dir}/{module_name}"

        design_filename = f"{module_name}.sv"
        testbench_filename = f"{module_name}_tb.sv"

        # Define .env key-value pairs with relative paths
        env_lines = [
            f"PROJECT_DIR={module_relative_path}",
            f"DESIGN_FILE={design_filename}",
            f"TESTBENCH_FILE={testbench_filename}",
            f"TOP_MODULE={module_name}_tb",
            f"VCD_FILE={module_name}_tb.vcd",
        ]

        # Join lines with newlines for .env content
        env_content = "\n".join(env_lines)

        return {"env_content": env_content, "error": ""}
    except Exception as error:
        return {
            "env_content": "",
            "error": f"Environment file generation error: {str(error)}",
        }


@tool
def save_code_tool(
    design_code: str,
    testbench_code: str,
    env_content: str,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """
    Save SystemVerilog code and .env file to a module-specific folder.

    Args:
        design_code: Generated design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        output_dir: Base directory for saving files (default: 'output').

    Returns:
        Dict:
            Contains list of status messages, saved file paths,
            and any error message.
    """
    try:
        messages = []
        saved_files = {}

        module_match = re.search(r"module\s+(\w+)", design_code)
        module_name = (
            module_match.group(1) if module_match else "generated_module"
        )

        # Create module-specific directory
        module_dir = os.path.join(output_dir, module_name)
        os.makedirs(module_dir, exist_ok=True)

        # Save design code
        design_filename = f"{module_name}.sv"
        design_filepath = os.path.join(module_dir, design_filename)
        with open(design_filepath, "w") as design_file:
            design_file.write(design_code)
        messages.append(f"Design saved to {design_filepath}")
        saved_files["design_file"] = design_filepath

        # Save testbench code if available
        testbench_filename = f"{module_name}_tb.sv"
        testbench_filepath = os.path.join(module_dir, testbench_filename)
        if testbench_code:
            with open(testbench_filepath, "w") as testbench_file:
                testbench_file.write(testbench_code)
            messages.append(f"Testbench saved to {testbench_filepath}")
            saved_files["testbench_file"] = testbench_filepath
        else:
            messages.append("No testbench code to save")

        # Save .env file
        env_filepath = os.path.join(module_dir, ".env")
        if env_content:
            with open(env_filepath, "w") as env_file:
                env_file.write(env_content)
            messages.append(f"Environment file saved to {env_filepath}")
            saved_files["env_file"] = env_filepath
        else:
            messages.append("No .env file content to save")

        return {
            "messages": messages,
            "error": "",
            "module_dir": module_dir,
            "saved_files": saved_files,
            "module_name": module_name,
        }
    except Exception as error:
        return {
            "messages": [],
            "error": f"Error saving code: {str(error)}",
            "module_dir": "",
            "saved_files": {},
            "module_name": "",
        }


@tool
def run_simulation_tool(
    target_dir: str, strip_lines: bool = True
) -> Dict[str, Any]:
    """
    Run Verilator simulation using Docker Compose.

    Args:
        target_dir:
            Directory containing the .env file and SystemVerilog files.
        strip_lines: Whether to strip first and last lines from output.

    Returns:
        Dict: Contains success status, return code, and any error message.
    """
    try:
        # Ensure target directory ends with separator for run_docker_compose
        if not target_dir.endswith(os.sep):
            target_dir += os.sep

        print(f"Starting Verilator simulation in: {target_dir}")

        # Run the simulation
        return_code = run_docker_compose(
            target_dir=target_dir, strip_lines=strip_lines
        )

        success = return_code == 0

        if success:
            message = "Verilator simulation completed successfully"
        else:
            message = (
                f"Verilator simulation failed with return code: {return_code}"
            )

        return {
            "success": success,
            "return_code": return_code,
            "message": message,
            "error": ""
            if success
            else f"Simulation failed (exit code: {return_code})",
        }

    except Exception as error:
        return {
            "success": False,
            "return_code": -1,
            "message": f"Error running simulation: {str(error)}",
            "error": f"Simulation tool error: {str(error)}",
        }


class SystemVerilogCodeGenerator:
    def __init__(self):
        """
        Initialize SystemVerilog code generator with LLM.

        Generates SystemVerilog code and testbench per standards.
        """
        self.tools = [
            load_saved_code_tool,
            generate_env_file_tool,
            save_code_tool,
            run_simulation_tool,
        ]
        self.llm = init_chat_model(LLM_MODEL).bind_tools(self.tools)
        self.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", LLM_INSTRUCTIONS),
            ("human", "{user_request}"),
        ])
        self.graph = self._create_graph()

    def _create_graph(self):
        """
        Create LangGraph workflow for code generation and simulation.

        Returns:
            StateGraph: Compiled workflow with nodes and edges.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("generate_env", self._call_generate_env_tool)
        workflow.add_node("save_code", self._call_save_code_tool)
        workflow.add_node("run_simulation", self._call_run_simulation_tool)

        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "generate_env")
        workflow.add_edge("generate_env", "save_code")
        workflow.add_edge("save_code", "run_simulation")
        workflow.add_edge("run_simulation", END)

        return workflow.compile()

    def _generate_code(self, state: AgentState) -> AgentState:
        """
        Generate SystemVerilog code and testbench from user request.

        Args:
            state: Current state with user request and data.

        Returns:
            AgentState: Updated with generated code, testbench, messages.
        """
        try:
            messages = [
                {"role": "system", "content": LLM_INSTRUCTIONS},
                {"role": "user", "content": state["user_request"]},
            ]
            response = self.llm.invoke(messages)
            content = response.content

            # Extract code blocks
            design_code = ""
            testbench_code = ""

            # Look for design code block
            design_match = re.search(
                r"```systemverilog\s+design\s*\n(.*?)```", content, re.DOTALL
            )
            if design_match:
                design_code = design_match.group(1).strip()

            # Look for testbench code block
            testbench_match = re.search(
                r"```systemverilog\s+testbench\s*\n(.*?)```",
                content,
                re.DOTALL,
            )
            if testbench_match:
                testbench_code = testbench_match.group(1).strip()

            if not design_code and not testbench_code:
                code_blocks = re.findall(
                    r"```systemverilog\s*\n(.*?)```", content, re.DOTALL
                )
                if len(code_blocks) >= 2:
                    design_code = code_blocks[0].strip()
                    testbench_code = code_blocks[1].strip()
                elif len(code_blocks) == 1:
                    design_code = code_blocks[0].strip()
                    state["error"] = "Testbench not provided by LLM"
                    return state
                else:
                    state["error"] = "No valid SystemVerilog code found"
                    return state
            elif not design_code:
                state["error"] = "Design code not found"
                return state
            elif not testbench_code:
                state["error"] = "Testbench code not found"
                return state

            state["generated_code"] = design_code
            state["testbench_code"] = testbench_code
            state["messages"].append(
                AIMessage(
                    content=f"Generated SystemVerilog design and testbench "
                    f"for: {state['user_request']}"
                )
            )
        except Exception as error:
            state["error"] = f"Code generation error: {str(error)}"
        return state

    def _call_generate_env_tool(self, state: AgentState) -> AgentState:
        """
        Call the generate_env_file_tool with the generated code.

        Args:
            state: Current state with generated code and data.

        Returns:
            AgentState: Updated with .env file content from tool.
        """
        try:
            result = generate_env_file_tool.invoke({
                "generated_code": state["generated_code"],
                "output_dir": state["output_dir"],
            })
            state["env_content"] = result["env_content"]
            if result["error"]:
                state["error"] = result["error"]
            else:
                module_name = (
                    re.search(
                        r"module\s+(\w+)", state["generated_code"]
                    ).group(1)
                    if re.search(r"module\s+(\w+)", state["generated_code"])
                    else "generated_module"
                )
                state["messages"].append(
                    AIMessage(
                        content=f"Generated .env file for {module_name}"
                    )
                )
        except Exception as error:
            state["error"] = f"Error calling env tool: {str(error)}"
        return state

    def _call_save_code_tool(self, state: AgentState) -> AgentState:
        """
        Call the save_code_tool with generated code and env content.

        Args:
            state:
                Current state with generated code, testbench, and env content.

        Returns:
            AgentState: Updated with save status messages from tool.
        """
        try:
            result = save_code_tool.invoke({
                "design_code": state["generated_code"],
                "testbench_code": state["testbench_code"],
                "env_content": state["env_content"],
                "output_dir": state["output_dir"],
            })
            for message in result["messages"]:
                state["messages"].append(AIMessage(content=message))
            if result["error"]:
                state["error"] = result["error"]
            else:
                # Store the module directory and saved files for later use
                state["module_dir"] = result.get("module_dir", "")
                state["saved_files"] = result.get("saved_files", {})
        except Exception as error:
            state["error"] = f"Error calling save code tool: {str(error)}"
        return state

    def _call_run_simulation_tool(self, state: AgentState) -> AgentState:
        """
        Call the run_simulation_tool with the module directory.

        Args:
            state: Current state with module directory and data.

        Returns:
            AgentState: Updated with simulation results from tool.
        """
        try:
            result = run_simulation_tool.invoke({
                "target_dir": state["module_dir"],
                "strip_lines": True,
            })
            state["messages"].append(AIMessage(content=result["message"]))
            if result["error"]:
                state["error"] = result["error"]
        except Exception as error:
            state["error"] = f"Error calling simulation tool: {str(error)}"
        return state

    def load_existing_code(self, module_dir: str) -> Dict[str, Any]:
        """
        Load existing SystemVerilog code from saved files.

        Args:
            module_dir: Directory containing the saved SystemVerilog files.

        Returns:
            Dict: Contains loaded code and any error messages.
        """
        try:
            result = load_saved_code_tool.invoke({"module_dir": module_dir})
            return result
        except Exception as error:
            return {
                "design_code": "",
                "testbench_code": "",
                "env_content": "",
                "module_name": "",
                "error": f"Error loading existing code: {str(error)}",
            }

    def run_simulation_on_existing(self, module_dir: str) -> Dict[str, Any]:
        """
        Run simulation on existing saved code.

        Args:
            module_dir: Directory containing the saved SystemVerilog files.

        Returns:
            Dict: Contains simulation results and any error messages.
        """
        try:
            # First verify the files exist
            if not os.path.exists(module_dir):
                return {
                    "success": False,
                    "error": f"Module directory does not exist: {module_dir}",
                }

            # Check for required files
            env_file = os.path.join(module_dir, ".env")
            if not os.path.exists(env_file):
                return {
                    "success": False,
                    "error": f"Environment file not found: {env_file}",
                }

            # Run simulation
            result = run_simulation_tool.invoke({
                "target_dir": module_dir,
                "strip_lines": True,
            })

            return result

        except Exception as error:
            return {
                "success": False,
                "return_code": -1,
                "message": f"Error running simulation: {str(error)}",
                "error": str(error),
            }

    def generate(
        self,
        user_request: str,
        save_file: bool = False,
        output_dir: str = "output",
    ) -> Dict[str, Any]:
        """Generate SystemVerilog design, testbench, .env file.

        Args:
            user_request: Desired SystemVerilog module description.
            save_file: Save generated code to files (default: False).
            output_dir: Directory for saving files (default: 'output').

        Returns:
            Dict:
                Contains success, design_code, testbench_code, env_content,
                error, messages, and saved_files.
        """
        initial_state = {
            "user_request": user_request,
            "generated_code": "",
            "testbench_code": "",
            "env_content": "",
            "messages": [HumanMessage(content=user_request)],
            "error": "",
            "output_dir": output_dir,
            "module_dir": "",
            "saved_files": {},
        }

        if save_file:
            # Ensure output directory exists
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as error:
                initial_state["error"] = (
                    f"Failed to create output directory: {str(error)}"
                )
                return {
                    "success": False,
                    "design_code": "",
                    "testbench_code": "",
                    "env_content": "",
                    "error": initial_state["error"],
                    "messages": initial_state["messages"],
                    "saved_files": {},
                }
            result = self.graph.invoke(initial_state)
        else:
            # Skip save_code and run_simulation nodes if not saving files
            workflow = StateGraph(AgentState)
            workflow.add_node("generate_code", self._generate_code)
            workflow.add_node("generate_env", self._call_generate_env_tool)
            workflow.add_edge(START, "generate_code")
            workflow.add_edge("generate_code", "generate_env")
            workflow.add_edge("generate_env", END)
            result = workflow.compile().invoke(initial_state)

        return {
            "success": not bool(result["error"]),
            "design_code": result["generated_code"],
            "testbench_code": result["testbench_code"],
            "env_content": result["env_content"],
            "error": result["error"],
            "messages": result["messages"],
            "saved_files": result.get("saved_files", {}),
            "module_dir": result.get("module_dir", ""),
        }


if __name__ == "__main__":
    generator = SystemVerilogCodeGenerator()

    print("SystemVerilog Code Generator Graph:")
    print(generator.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60)

    # Example requests
    test_requests = [
        "Create a simple 4-bit counter module",
        # "Generate a 2-to-1 multiplexer with enable signal",
        # "Create a D flip-flop with asynchronous reset",
    ]

    for request in test_requests:
        print(f"\nRequest: {request}")
        print("=" * 60)

        result = generator.generate(
            request,
            save_file=True,
            output_dir="sv_output",
        )

        if result["success"]:
            print("✅ Design and testbench generation successful")
            print(f"📁 Module directory: {result['module_dir']}")

            # Display saved files
            if result["saved_files"]:
                print("📄 Saved files:")
                for file_type, file_path in result["saved_files"].items():
                    print(f"  - {file_type}: {file_path}")

            # Test loading existing code
            print("\n🔄 Testing code loading from saved files...")
            loaded_result = generator.load_existing_code(result["module_dir"])
            if loaded_result["error"]:
                print(
                    f"❌ Error loading saved code: {loaded_result['error']}"
                )
            else:
                print("✅ Successfully loaded saved code")
                print(f"📦 Module name: {loaded_result['module_name']}")

            # Run simulation on existing code
            print("\n🔄 Running simulation on saved code...")
            sim_result = generator.run_simulation_on_existing(
                result["module_dir"]
            )
            if sim_result.get("success", False):
                print("✅ Simulation completed successfully")
            else:
                print(
                    f"❌ Simulation failed: {
                        sim_result.get('message', 'Unknown error')
                    }"
                )
        else:
            print(f"❌ Error: {result['error']}")

        print("\n" + "=" * 60)
