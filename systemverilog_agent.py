"""SystemVerilog code generator for creating and testing designs.

This module provides a SystemVerilogCodeGenerator class to generate
SystemVerilog design and testbench code, create .env files, save
generated files, and run Verilator simulations using a LangGraph
workflow. It supports retry on failure and regeneration on success.
Includes cleanup on failure and limits retries/regenerations.

Attributes:
    LLM_MODEL: Model identifier for the language model.
    LLM_INSTRUCTIONS: Instructions for SystemVerilog code generation.
"""

import os
import re
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from systemverilog_agent_handlers import SVHandlers

load_dotenv()

# LLM_MODEL = "groq:llama-3.3-70b-versatile"
# LLM_MODEL = "groq:deepseek-r1-distill-llama-70b"
LLM_MODEL = "groq:qwen/qwen3-32b"

LLM_INSTRUCTIONS = """
You are an expert SystemVerilog code generator.
Generate clean, well-structured SystemVerilog code based on user requirements.
Adapt the code for Verilator.

**Important Path Generation Requirements**:
  - Based on the user request, generate an appropriate directory path
  - Use format: "Chapter_X_examples/example_Y_design_name" where:
    - X is the chapter number (infer from context or use appropriate number)
    - Y is the example number within that chapter
    - design_name is the module name or descriptive name
  - Examples:
    - "Chapter_1_examples/example_1_four_bit_counter"
  - If no specific chapter is mentioned, use Chapter_1_examples
  - Make the path descriptive and logical based on the design complexity

**Coding Standards**:
  - Maximum 80 chars per line.
  - Use 2-space indent.
  - Include clear, relevant comments.
  - Use meaningful signal names.
  - Follow SystemVerilog best practices.
  - Add a comment with just the file name (// test.sv)
  - Add comments to explain the code for beginners.
  - Adapt the code for Verilator

**Requirements**:
  1. **Directory Path**:
     - Generate an appropriate directory path
     - Use format: Chapter_X_examples/example_Y_design_name
  2. **Design Module**:
     - Create a complete, compilable SystemVerilog module.
  3. **Testbench Module**:
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
  - Include informative $display statements in the testbench:
    - Print a header: `$display("=== <Module Name> Testbench Started ===");`
    - Print test phase messages:
    `$display("Testing <specific_functionality>...");`
    - Print input values: `$display("Applying inputs: <input_description>");`
    - Print expected vs actual:
    `$display("Expected: %d, Got: %d", expected, actual);`
    - Print pass/fail status:
    `$display("Test <test_name>: %s", pass ? "PASSED" : "FAILED");`
    - Print a summary: `$display("=== Testbench Completed ===");`
  - Use $time in messages where relevant:
  `$display("Time %0t: <message>", $time);`
  - Include error messages for failed assertions or unexpected behavior
  - Make messages clear and descriptive to help with debugging

**Output Format**:
  - Provide exactly three items in order:
    1. **Directory Path**: A single line with the suggested directory path
       Format: DIRECTORY_PATH: Chapter_X_examples/example_Y_design_name
    2. **Design Code Block**:
       ```systemverilog design
       <Design module code here>
       ```
    3. **Testbench Code Block**:
       ```systemverilog testbench
       <Testbench module code here>
       ```
  - Ensure all three items are present and correctly formatted.
  - Do not include other code blocks or markers.
"""


class AgentState(TypedDict):
    """State for SystemVerilog code generation agent.

    Attributes:
        user_request: User request for SystemVerilog code.
        generated_code: Generated SystemVerilog design code.
        testbench_code: Generated testbench code.
        env_content: Generated .env file content.
        generated_path: LLM-generated directory path for the module.
        messages: List of conversation messages.
        error: Any error messages during code generation.
        output_dir: Base directory for saving files.
        module_dir: Full path to the module-specific directory.
        saved_files: Dictionary tracking saved file paths.
        user_retry_confirmed: Whether user confirmed retry via prompt.
        cleanup_performed: Whether cleanup was performed on failure.
        user_regenerate_confirmed:
            Whether user confirmed regeneration after success.
        retry_count: Number of retry attempts made.
        regeneration_count: Number of regeneration attempts made.
        max_retries: Maximum allowed retries.
        max_regenerations: Maximum allowed regenerations.
        cleanup_on_retry: Whether to clean up files before retrying.
    """

    user_request: str
    generated_code: str
    testbench_code: str
    env_content: str
    generated_path: str
    messages: Annotated[list, add_messages]
    error: str
    output_dir: str
    module_dir: str
    saved_files: Dict[str, str]
    user_retry_confirmed: bool
    cleanup_performed: bool
    user_regenerate_confirmed: bool
    retry_count: int
    regeneration_count: int
    max_retries: int
    max_regenerations: int
    cleanup_on_retry: bool


class SystemVerilogCodeGenerator:
    def __init__(self):
        """Initializes the SystemVerilog code generator with LLM and tools.

        Sets up the language model, prompt, and workflow for generating
        SystemVerilog code and testbenches per standards.
        """
        self.llm = init_chat_model(LLM_MODEL)
        self.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", LLM_INSTRUCTIONS),
            ("human", "{user_request}"),
        ])
        self.graph = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        """Creates a LangGraph workflow for SystemVerilog code generation."""
        workflow = StateGraph(AgentState)

        # Linear workflow edges
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "generate_env")
        workflow.add_edge("generate_env", "save_code")
        workflow.add_edge("save_code", "run_simulation")

        # Add nodes
        workflow.add_node("generate_code", self.generate_systemverilog)
        workflow.add_node("generate_env", SVHandlers.create_env_file)
        workflow.add_node("save_code", SVHandlers.save_generated_files)
        workflow.add_node("run_simulation", SVHandlers.execute_simulation)
        workflow.add_node("cleanup_files", SVHandlers.cleanup_on_failure)
        workflow.add_node("simulation_result", self.simulation_result)

        # Conditional routing after simulation
        def route_simulation_result(state: AgentState) -> str:
            """Routes to simulation_result based on user choice."""
            return "simulation_result"

        workflow.add_conditional_edges(
            "run_simulation",
            route_simulation_result,
            {"simulation_result": "simulation_result"},
        )

        # Conditional routing after handling simulation result
        def route_handle_result(state: AgentState) -> str:
            """Routes based on user decisions."""
            if state.get("retry_count", 0) >= state.get("max_retries", 5):
                return "retry_limit_reached"
            if state.get("regeneration_count", 0) >= state.get(
                "max_regenerations", 5
            ):
                return "regeneration_limit_reached"
            if state.get("cleanup_on_retry", False):
                return "cleanup_files"
            if state.get("user_retry_confirmed", False) or state.get(
                "user_regenerate_confirmed", False
            ):
                return "generate_code"
            return "user_exit"

        workflow.add_conditional_edges(
            "simulation_result",
            route_handle_result,
            {
                "cleanup_files": "cleanup_files",
                "generate_code": "generate_code",
                "user_exit": END,
                "retry_limit_reached": END,
                "regeneration_limit_reached": END,
            },
        )

        # After cleanup, go back to generate_code if retrying
        workflow.add_edge("cleanup_files", "generate_code")

        return workflow.compile()

    def generate_systemverilog(self, state: AgentState) -> AgentState:
        """Generates SystemVerilog design and testbench code.

        Uses LLM to generate code per standards, extracting directory path,
        design and testbench code blocks.

        Args:
            state: Agent state with user request and other data.

        Returns:
            AgentState:
                Updated with generated code, testbench, path, messages.

        Raises:
            Exception: If code generation or extraction fails.
        """
        try:
            messages = [
                {"role": "system", "content": LLM_INSTRUCTIONS},
                {"role": "user", "content": state["user_request"]},
            ]
            response = self.llm.invoke(messages)
            content = response.content

            # Extract directory path
            generated_path = ""
            path_match = re.search(
                r"DIRECTORY_PATH:\s*(.+)", content, re.IGNORECASE
            )
            if path_match:
                generated_path = path_match.group(1).strip()

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

            # Fallback to generic code blocks if specific ones not found
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

            # Use generated path or fallback to default
            if not generated_path:
                # Extract module name for fallback path
                module_match = re.search(r"module\s+(\w+)", design_code)
                module_name = (
                    module_match.group(1)
                    if module_match
                    else "generated_module"
                )
                generated_path = (
                    f"Chapter_1_examples/example_1__{module_name}"
                )

            state["generated_code"] = design_code
            state["testbench_code"] = testbench_code
            state["generated_path"] = generated_path
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Generated SystemVerilog design and testbench for: "
                        f"{state['user_request']} (Path: {generated_path})"
                    )
                )
            )
        except Exception as error:
            state["error"] = f"Code generation error: {str(error)}"
        return state

    def simulation_result(self, state: AgentState) -> AgentState:
        """Handles simulation result, prompting for retry or regeneration.

        Combines retry on failure and regeneration on success logic. Prompts
        user to retry on failure with optional cleanup or regenerate on
        success. Increments retry_count or regeneration_count and checks
        limits.

        Args:
            state: Agent state with simulation results.

        Returns:
            AgentState: Updated with user decisions, counts, and reset fields.
        """
        try:
            simulation_failed = (
                bool(state["error"])
                and "simulation" in state["error"].lower()
            )

            if simulation_failed:
                state["retry_count"] = state.get("retry_count", 0) + 1
                if state["retry_count"] >= state.get("max_retries", 5):
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"Maximum retries ({state['max_retries']}) "
                                f"reached."
                            )
                        )
                    )
                    state["user_retry_confirmed"] = False
                    state["user_regenerate_confirmed"] = False
                    return state

                cleanup_status = (
                    "Files cleaned up."
                    if state.get("cleanup_performed", False)
                    else "Cleanup may have failed."
                )
                print(f"\nSimulation failed: {state['error']}")
                print(f"Cleanup status: {cleanup_status}")
                print(
                    "Generated path: ",
                    f"{state.get('generated_path', 'Not set')}",
                )
                print(
                    f"Retry attempt {state['retry_count']} of "
                    f"{state['max_retries']}"
                )

                user_input = (
                    input(
                        "Press Enter to retry, 'c' to retry with cleanup, "
                        "or 'q' to quit: "
                    )
                    .strip()
                    .lower()
                )

                if user_input in ("", "enter"):
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"Retry {state['retry_count']} confirmed "
                                f"without cleanup."
                            )
                        )
                    )
                    state["user_retry_confirmed"] = True
                    state["user_regenerate_confirmed"] = False
                    state["cleanup_on_retry"] = False
                    # Reset code-related fields
                    state["generated_code"] = ""
                    state["testbench_code"] = ""
                    state["env_content"] = ""
                    state["generated_path"] = ""
                    state["error"] = ""
                    state["saved_files"] = {}
                    state["cleanup_performed"] = False
                    return state
                elif user_input == "c":
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"Retry {state['retry_count']} confirmed "
                                f"with cleanup."
                            )
                        )
                    )
                    state["user_retry_confirmed"] = True
                    state["user_regenerate_confirmed"] = False
                    state["cleanup_on_retry"] = True
                    # Reset code-related fields
                    state["generated_code"] = ""
                    state["testbench_code"] = ""
                    state["env_content"] = ""
                    state["generated_path"] = ""
                    state["error"] = ""
                    state["saved_files"] = {}
                    state["cleanup_performed"] = False
                    return state
                else:
                    state["messages"].append(
                        AIMessage(content="User chose to quit retry process.")
                    )
                    state["user_retry_confirmed"] = False
                    state["user_regenerate_confirmed"] = False
                    state["cleanup_on_retry"] = False
                    return state
            else:
                state["regeneration_count"] = (
                    state.get("regeneration_count", 0) + 1
                )
                if state["regeneration_count"] >= state.get(
                    "max_regenerations", 5
                ):
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"Maximum regenerations "
                                f"({state['max_regenerations']}) reached."
                            )
                        )
                    )
                    state["user_regenerate_confirmed"] = False
                    state["user_retry_confirmed"] = False
                    return state

                print("\n✅ Simulation successful!")
                print(
                    "Generated path: ",
                    f"{state.get('generated_path', 'Not set')}",
                )
                print(f"Module directory: {state['module_dir']}")
                print(f"Generated files: {list(state['saved_files'].keys())}")
                print(
                    f"Regeneration attempt {state['regeneration_count']} of "
                    f"{state['max_regenerations']}"
                )

                user_input = (
                    input("Press Enter to regenerate code or 'q' to quit: ")
                    .strip()
                    .lower()
                )

                if user_input in ("", "enter"):
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"User confirmed regeneration "
                                f"{state['regeneration_count']} after "
                                f"successful simulation."
                            )
                        )
                    )
                    state["user_regenerate_confirmed"] = True
                    state["user_retry_confirmed"] = False
                    state["cleanup_on_retry"] = False
                    # Reset only code-related fields, keep directory structure
                    state["generated_code"] = ""
                    state["testbench_code"] = ""
                    state["env_content"] = ""
                    state["generated_path"] = ""
                    state["error"] = ""
                    return state
                else:
                    state["messages"].append(
                        AIMessage(
                            content="User chose to quit regeneration process."
                        )
                    )
                    state["user_regenerate_confirmed"] = False
                    state["user_retry_confirmed"] = False
                    state["cleanup_on_retry"] = False
                    return state
        except Exception as error:
            state["error"] = f"Simulation result handling error: {str(error)}"
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Error in handling simulation result: {str(error)}"
                    )
                )
            )
            state["user_retry_confirmed"] = False
            state["user_regenerate_confirmed"] = False
            state["cleanup_on_retry"] = False
            return state

    def generate(
        self,
        user_request: str,
        output_dir: str = ".",
        recursion_limit: int = 50,
        max_retries: int = 5,
        max_regenerations: int = 5,
    ) -> Dict[str, Any]:
        """Generates SystemVerilog design, testbench, and .env file.

        Saves files and runs simulation with interactive retry on failure
        and regeneration on success. Includes cleanup on failure.
        Configurable recursion limit and max retries/regenerations.
        Uses LLM-generated directory paths.

        Args:
            user_request: Description of the desired SystemVerilog module.
            output_dir: Base directory for saving files (default: '.').
            recursion_limit:
                Maximum recursion limit for LangGraph (default: 50).
            max_retries: Maximum number of retry attempts (default: 5).
            max_regenerations:
                Maximum number of regeneration attempts (default: 5).

        Returns:
            Dict:
                Contains success, design_code, testbench_code, env_content,
                generated_path, error, messages, saved_files, module_dir,
                cleanup_performed, user_regenerate_confirmed,
                cleanup_on_retry.

        Raises:
            OSError: If output directory creation fails.
            Exception: For errors in code generation or simulation.
        """
        initial_state = {
            "user_request": user_request,
            "generated_code": "",
            "testbench_code": "",
            "env_content": "",
            "generated_path": "",
            "messages": [HumanMessage(content=user_request)],
            "error": "",
            "output_dir": output_dir,
            "module_dir": "",
            "saved_files": {},
            "user_retry_confirmed": False,
            "cleanup_performed": False,
            "user_regenerate_confirmed": False,
            "retry_count": 0,
            "regeneration_count": 0,
            "max_retries": max_retries,
            "max_regenerations": max_regenerations,
            "cleanup_on_retry": False,
        }

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
                "generated_path": "",
                "error": initial_state["error"],
                "messages": initial_state["messages"],
                "saved_files": {},
                "module_dir": "",
                "cleanup_performed": False,
                "user_regenerate_confirmed": False,
                "retry_count": 0,
                "regeneration_count": 0,
                "cleanup_on_retry": False,
            }

        # Always use the full workflow (save files and run simulation)
        result = self.graph.invoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )

        return {
            "success": not bool(result["error"]),
            "design_code": result["generated_code"],
            "testbench_code": result["testbench_code"],
            "env_content": result["env_content"],
            "generated_path": result.get("generated_path", ""),
            "error": result["error"],
            "messages": result["messages"],
            "saved_files": result.get("saved_files", {}),
            "module_dir": result.get("module_dir", ""),
            "cleanup_performed": result.get("cleanup_performed", False),
            "user_regenerate_confirmed": result.get(
                "user_regenerate_confirmed", False
            ),
            "retry_count": result.get("retry_count", 0),
            "regeneration_count": result.get("regeneration_count", 0),
            "cleanup_on_retry": result.get("cleanup_on_retry", False),
        }

    def run_interactive_mode(self):
        """Runs the generator in interactive mode."""
        print("SystemVerilog Code Generator - Interactive Mode")
        print("Type 'q' to stop")
        print("=" * 60)

        while True:
            try:
                user_request = input(
                    "\nEnter your SystemVerilog request (type 'q' to exit):\n"
                )

                if user_request.lower() == "q":
                    print("Goodbye!")
                    break

                if not user_request.strip():
                    print("Please enter a valid request.")
                    continue

                print(f"\nProcessing: {user_request}")
                print("=" * 60)

                result = self.generate(
                    user_request,
                    output_dir=".",
                    recursion_limit=200,
                    max_retries=10,
                    max_regenerations=10,
                )

                self._print_result(result)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def run_batch_mode(self, test_requests=None):
        """Runs the generator in batch mode with predefined requests."""
        if test_requests is None:
            test_requests = [
                "Create a simple 4-bit counter module, name it "
                "'four_bit_counter' this is for chapter 1, example 1",
                "Generate a 2-to-1 multiplexer with enable signal, name it "
                "'mux_2to1', chapter 2, example 2",
                "Create a D flip-flop with asynchronous reset, name the "
                "module 'dff', chapter 3, example 3",
            ]

        print("SystemVerilog Code Generator - Batch Mode")
        print("=" * 60)

        for i, request in enumerate(test_requests, 1):
            print(f"\n[{i}/{len(test_requests)}] Request: {request}")
            print("=" * 60)

            result = self.generate(
                request,
                output_dir=".",
                recursion_limit=200,
                max_retries=10,
                max_regenerations=10,
            )

            self._print_result(result)

    def _print_result(self, result):
        """Prints the result of a generation request."""
        print(f"Success: {result['success']}")
        print(f"Module directory: {result['module_dir'] or 'Not set'}")
        print(f"Cleanup performed: {result['cleanup_performed']}")
        print(
            f"Regeneration confirmed: {result['user_regenerate_confirmed']}"
        )
        print(f"Retry count: {result['retry_count']}")
        print(f"Regeneration count: {result['regeneration_count']}")
        print(f"Cleanup on retry: {result['cleanup_on_retry']}")

        if result["error"]:
            print(f"Error: {result['error']}")

        if result["success"]:
            print("✅ Design and testbench generation successful")
            if result["saved_files"]:
                print("📄 Saved files:")
                for file_type, file_path in result["saved_files"].items():
                    print(f"  - {file_type}: {file_path}")
        else:
            print(f"❌ Generation failed: {result['error']}")
            if result["cleanup_performed"]:
                print("🧹 Files were cleaned up after failure")
            print(
                "Files have been cleaned up, ready for fresh retry if needed"
            )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    generator = SystemVerilogCodeGenerator()

    print("SystemVerilog Code Generator Graph:")
    print(generator.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60)

    print("\nChoose mode: \n1. Interactive \n2. Batch \n3. Custom batch")
    mode = input("\nEnter choice (1-3): ")

    if mode.lower() == "1":
        generator.run_interactive_mode()
    elif mode.lower() == "2":
        generator.run_batch_mode()
    elif mode.lower() == "3":
        # Custom batch mode
        requests = []
        print("Enter requests (empty line to finish):")
        while True:
            request = input("Request: ")
            if not request.strip():
                break
            requests.append(request)

        if requests:
            generator.run_batch_mode(requests)
        else:
            print("No requests provided.")
    else:
        print("Invalid mode selected.")
