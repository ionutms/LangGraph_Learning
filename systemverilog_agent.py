"""SystemVerilog code generator for creating and testing designs.

This module provides a SystemVerilogCodeGenerator class to generate
SystemVerilog design and testbench code, create .env files, save
generated files, and run Verilator simulations using a LangGraph
workflow. It includes tools for loading and simulating existing code
and supports interactive retry on simulation failure with user input.
Now includes cleanup functionality to remove files on simulation failure
and regeneration capability after successful simulation. Adds recursion
limit and max retries/regenerations to prevent infinite loops.

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

from systemverilog_agent_tools import (
    cleanup_files_tool,
    generate_env_file_tool,
    run_simulation_tool,
    save_code_tool,
)

load_dotenv()

LLM_MODEL = "groq:llama-3.3-70b-versatile"

LLM_INSTRUCTIONS = """
You are an expert SystemVerilog code generator.
Generate clean, well-structured SystemVerilog code based on user requirements.
Adapt the code for Verilator.

**Coding Standards**:
  - Maximum 80 chars per line.
  - Use 2-space indent.
  - Include clear, relevant comments.
  - Use meaningful signal names.
  - Follow SystemVerilog best practices.
  - Add a comment which contains just the file name (// test.sv)
  - Add comments to explain the code like for an absolute.
  - Adapt the code for Verilator

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
    """State for SystemVerilog code generation agent.

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
        user_retry_confirmed: Whether user confirmed retry via prompt.
        cleanup_performed: Whether cleanup was performed on failure.
        user_regenerate_confirmed: Whether user confirmed regeneration
            after success.
        retry_count: Number of retry attempts made.
        regeneration_count: Number of regeneration attempts made.
        max_retries: Maximum allowed retries.
        max_regenerations: Maximum allowed regenerations.
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
    user_retry_confirmed: bool
    cleanup_performed: bool
    user_regenerate_confirmed: bool
    retry_count: int
    regeneration_count: int
    max_retries: int
    max_regenerations: int


class SystemVerilogCodeGenerator:
    def __init__(self):
        """Initializes the SystemVerilog code generator with LLM and tools.

        Sets up the language model, prompt, and workflow for generating
        SystemVerilog code and testbenches per standards.
        """
        self.tools = [
            cleanup_files_tool,
            generate_env_file_tool,
            run_simulation_tool,
            save_code_tool,
        ]
        self.llm = init_chat_model(LLM_MODEL).bind_tools(self.tools)
        self.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", LLM_INSTRUCTIONS),
            ("human", "{user_request}"),
        ])
        self.graph = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        """Creates a LangGraph workflow for SystemVerilog code generation."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate_code", self.generate_systemverilog)
        workflow.add_node("generate_env", self.create_env_file)
        workflow.add_node("save_code", self.save_generated_files)
        workflow.add_node("run_simulation", self.execute_simulation)
        workflow.add_node("cleanup_files", self.cleanup_on_failure)
        workflow.add_node("retry_on_failure", self.retry_on_failure)
        workflow.add_node("regenerate_on_success", self.regenerate_on_success)

        # Linear workflow edges
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "generate_env")
        workflow.add_edge("generate_env", "save_code")
        workflow.add_edge("save_code", "run_simulation")

        # Conditional routing after simulation
        def route_simulation_result(state: AgentState) -> str:
            """Routes based on simulation success/failure."""
            simulation_failed = (
                bool(state["error"])
                and "simulation" in state["error"].lower()
            )
            return (
                "simulation_failed"
                if simulation_failed
                else "simulation_success"
            )

        workflow.add_conditional_edges(
            "run_simulation",
            route_simulation_result,
            {
                "simulation_failed": "cleanup_files",
                "simulation_success": "regenerate_on_success",
            },
        )

        # After cleanup, go to retry decision
        workflow.add_edge("cleanup_files", "retry_on_failure")

        # Conditional routing after retry decision
        def route_retry_decision(state: AgentState) -> str:
            """Routes based on user retry confirmation and retry limit."""
            if state.get("retry_count", 0) >= state.get("max_retries", 5):
                return "retry_limit_reached"
            return (
                "user_retry_confirmed"
                if state.get("user_retry_confirmed", False)
                else "user_exit"
            )

        workflow.add_conditional_edges(
            "retry_on_failure",
            route_retry_decision,
            {
                "user_retry_confirmed": "generate_code",
                "user_exit": END,
                "retry_limit_reached": END,
            },
        )

        # Conditional routing after regeneration decision
        def route_regenerate_decision(state: AgentState) -> str:
            """Routes based on user regeneration confirmation and limit."""
            if state.get("regeneration_count", 0) >= state.get(
                "max_regenerations", 5
            ):
                return "regeneration_limit_reached"
            return (
                "user_regenerate_confirmed"
                if state.get("user_regenerate_confirmed", False)
                else "user_exit"
            )

        workflow.add_conditional_edges(
            "regenerate_on_success",
            route_regenerate_decision,
            {
                "user_regenerate_confirmed": "generate_code",
                "user_exit": END,
                "regeneration_limit_reached": END,
            },
        )

        return workflow.compile()

    def generate_systemverilog(self, state: AgentState) -> AgentState:
        """Generates SystemVerilog design and testbench code.

        Uses LLM to generate code per standards, extracting design and
        testbench code blocks.

        Args:
            state: Agent state with user request and other data.

        Returns:
            AgentState: Updated with generated code, testbench, messages.

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
                    content=(
                        f"Generated SystemVerilog design and testbench "
                        f"for: {state['user_request']}"
                    )
                )
            )
        except Exception as error:
            state["error"] = f"Code generation error: {str(error)}"
        return state

    def create_env_file(self, state: AgentState) -> AgentState:
        """Generates .env file content using generate_env_file_tool.

        Extracts module name from code and creates .env with project config.

        Args:
            state: Agent state with generated code and output directory.

        Returns:
            AgentState: Updated with .env content and status messages.

        Raises:
            Exception: If .env file generation fails.
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
            state["error"] = f"Error generating .env file: {str(error)}"
        return state

    def save_generated_files(self, state: AgentState) -> AgentState:
        """Saves generated code and .env using save_code_tool.

        Stores design, testbench, and .env files in a module-specific dir.
        If module_dir already exists (from regeneration), uses the same
        directory.

        Args:
            state: Agent state with code, testbench, and .env content.

        Returns:
            AgentState: Updated with save status messages and file paths.

        Raises:
            Exception: If file saving fails due to I/O or other errors.
        """
        try:
            # Pass existing module_dir if available (for regeneration)
            save_params = {
                "design_code": state["generated_code"],
                "testbench_code": state["testbench_code"],
                "env_content": state["env_content"],
                "output_dir": state["output_dir"],
            }

            # If we have an existing module_dir, use it for regeneration
            if state.get("module_dir"):
                save_params["existing_module_dir"] = state["module_dir"]

            result = save_code_tool.invoke(save_params)

            for message in result["messages"]:
                state["messages"].append(AIMessage(content=message))
            if result["error"]:
                state["error"] = result["error"]
            else:
                # Store the module directory and saved files for later use
                state["module_dir"] = result.get("module_dir", "")
                state["saved_files"] = result.get("saved_files", {})
        except Exception as error:
            state["error"] = f"Error saving files: {str(error)}"
        return state

    def execute_simulation(self, state: AgentState) -> AgentState:
        """Executes Verilator simulation using run_simulation_tool.

        Runs simulation for saved SystemVerilog files in module directory.

        Args:
            state: Agent state with module directory and other data.

        Returns:
            AgentState: Updated with simulation results and messages.

        Raises:
            Exception: If simulation execution fails.
        """
        try:
            if not state["module_dir"]:
                state["error"] = "Module directory not set for simulation"
                state["messages"].append(
                    AIMessage(content="Error: Module directory not set")
                )
                return state
            result = run_simulation_tool.invoke({
                "target_dir": state["module_dir"],
                "strip_lines": True,
            })
            state["messages"].append(AIMessage(content=result["message"]))
            if result["error"]:
                state["error"] = result["error"]
        except Exception as error:
            state["error"] = f"Error executing simulation: {str(error)}"
        return state

    def cleanup_on_failure(self, state: AgentState) -> AgentState:
        """Cleans up generated files when simulation fails.

        Removes all generated files and the module directory to maintain
        a clean workspace before offering retry option.

        Args:
            state: Agent state with module directory and saved files info.

        Returns:
            AgentState: Updated with cleanup status and messages.
        """
        try:
            if not state["module_dir"]:
                state["messages"].append(
                    AIMessage(content="No module directory to clean up")
                )
                state["cleanup_performed"] = False
                return state

            result = cleanup_files_tool.invoke({
                "module_dir": state["module_dir"],
                "saved_files": state["saved_files"],
            })

            if result["success"]:
                state["messages"].append(
                    AIMessage(content=f"🧹 Cleanup: {result['message']}")
                )
                state["cleanup_performed"] = True
                # Clear the module_dir and saved_files since they're gone
                state["module_dir"] = ""
                state["saved_files"] = {}
            else:
                state["messages"].append(
                    AIMessage(content=f"❌ Cleanup failed: {result['error']}")
                )
                state["cleanup_performed"] = False
                # Don't clear module_dir if cleanup failed

        except Exception as error:
            state["error"] = f"Error during cleanup: {str(error)}"
            state["messages"].append(
                AIMessage(content=f"❌ Cleanup error: {str(error)}")
            )
            state["cleanup_performed"] = False

        return state

    def retry_on_failure(self, state: AgentState) -> AgentState:
        """Prompts user to retry the workflow if simulation fails.

        Asks for user input (Enter to retry, 'q' to quit) and resets
        state fields for retry if confirmed. Increments retry_count and
        checks against max_retries.

        Args:
            state: Agent state with simulation results.

        Returns:
            AgentState:
                Updated state with reset fields if retrying,
                or unchanged if not.
        """
        try:
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
                return state

            simulation_failed = (
                bool(state["error"])
                and "simulation" in state["error"].lower()
            )

            if simulation_failed:
                cleanup_status = (
                    "Files cleaned up."
                    if state.get("cleanup_performed", False)
                    else "Cleanup may have failed."
                )
                print(f"\nSimulation failed: {state['error']}")
                print(f"Cleanup status: {cleanup_status}")
                print(
                    f"Retry attempt {state['retry_count']} of "
                    f"{state['max_retries']}"
                )

                user_input = (
                    input("Press Enter to retry or 'q' to quit: ")
                    .strip()
                    .lower()
                )

                if user_input == "" or user_input == "enter":
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"Retry {state['retry_count']} confirmed "
                                f"after cleanup."
                            )
                        )
                    )
                    # Reset fields for retry
                    state["generated_code"] = ""
                    state["testbench_code"] = ""
                    state["env_content"] = ""
                    state["error"] = ""
                    state["saved_files"] = {}
                    state["cleanup_performed"] = False
                    state["user_regenerate_confirmed"] = False
                    state["user_retry_confirmed"] = True
                    return state
                else:
                    state["messages"].append(
                        AIMessage(content="User chose to quit retry process.")
                    )
                    state["user_retry_confirmed"] = False
                    return state
            else:
                state["messages"].append(
                    AIMessage(
                        content="No retry needed - simulation succeeded."
                    )
                )
                state["user_retry_confirmed"] = False
                return state
        except Exception as error:
            state["error"] = f"Retry decision error: {str(error)}"
            state["messages"].append(
                AIMessage(content=f"Error in retry decision: {str(error)}")
            )
            state["user_retry_confirmed"] = False
            return state

    def regenerate_on_success(self, state: AgentState) -> AgentState:
        """Prompts user to regenerate code after successful simulation.

        Asks for user input to regenerate the code with the same file and
        folder structure. Increments regeneration_count and checks against
        max_regenerations.

        Args:
            state: Agent state with successful simulation results.

        Returns:
            AgentState:
                Updated state with regeneration confirmation and
                preserved module directory for reuse.
        """
        try:
            # Increment regeneration count
            state["regeneration_count"] = (
                state.get("regeneration_count", 0) + 1
            )

            # Check if max regenerations reached
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
                return state

            # Check if simulation was successful
            simulation_succeeded = not bool(state["error"])

            if simulation_succeeded:
                print("\n✅ Simulation successful!")
                print(f"Module directory: {state['module_dir']}")
                print(f"Generated files: {list(state['saved_files'].keys())}")
                print(
                    f"Regeneration attempt {state['regeneration_count']} of "
                    f"{state['max_regenerations']}"
                )

                # Prompt user for regeneration
                user_input = (
                    input("Press Enter to regenerate code or 'q' to quit: ")
                    .strip()
                    .lower()
                )

                if user_input == "" or user_input == "enter":
                    state["messages"].append(
                        AIMessage(
                            content=(
                                f"User confirmed regeneration "
                                f"{state['regeneration_count']} after "
                                f"successful simulation."
                            )
                        )
                    )

                    # Reset only the code-related fields, keep directory
                    # structure
                    state["generated_code"] = ""
                    state["testbench_code"] = ""
                    state["env_content"] = ""
                    state["error"] = ""
                    # Keep module_dir and saved_files for reuse
                    state["user_regenerate_confirmed"] = True
                    state["user_retry_confirmed"] = False
                    return state
                else:
                    state["messages"].append(
                        AIMessage(
                            content="User chose to quit regeneration process."
                        )
                    )
                    state["user_regenerate_confirmed"] = False
                    return state
            else:
                state["messages"].append(
                    AIMessage(
                        content="No regeneration offered - simulation failed."
                    )
                )
                state["user_regenerate_confirmed"] = False
                return state
        except Exception as error:
            state["error"] = f"Regeneration decision error: {str(error)}"
            state["messages"].append(
                AIMessage(
                    content=f"Error in regeneration decision: {str(error)}"
                )
            )
            state["user_regenerate_confirmed"] = False
            return state

    def generate(
        self,
        user_request: str,
        output_dir: str = "output",
        recursion_limit: int = 50,
        max_retries: int = 5,
        max_regenerations: int = 5,
    ) -> Dict[str, Any]:
        """Generates SystemVerilog design, testbench, and .env file.

        Always saves files and runs simulation with interactive retry
        on simulation failure and regeneration option on success.
        Includes cleanup on failure. Configurable recursion limit and
        max retries/regenerations to prevent infinite loops.

        Args:
            user_request: Description of the desired SystemVerilog module.
            output_dir: Directory for saving files (default: 'output').
            recursion_limit: Maximum recursion limit for LangGraph
                (default: 50).
            max_retries: Maximum number of retry attempts (default: 5).
            max_regenerations: Maximum number of regeneration attempts
                (default: 5).

        Returns:
            Dict: Contains success, design_code, testbench_code, env_content,
                error, messages, saved_files, module_dir,
                cleanup_performed, and user_regenerate_confirmed.

        Raises:
            OSError: If output directory creation fails.
            Exception: For errors in code generation or simulation.
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
            "user_retry_confirmed": False,
            "cleanup_performed": False,
            "user_regenerate_confirmed": False,
            "retry_count": 0,
            "regeneration_count": 0,
            "max_retries": max_retries,
            "max_regenerations": max_regenerations,
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
                "error": initial_state["error"],
                "messages": initial_state["messages"],
                "saved_files": {},
                "module_dir": "",
                "cleanup_performed": False,
                "user_regenerate_confirmed": False,
                "retry_count": 0,
                "regeneration_count": 0,
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
        }


if __name__ == "__main__":
    generator = SystemVerilogCodeGenerator()

    print("SystemVerilog Code Generator Graph:")
    print(generator.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60)

    user_input = input(
        "Run demo test requests or provide you prompt? (y/n)\n"
    )

    # Example requests
    test_requests = [
        "Create a simple 4-bit counter module, name it 'four_bit_counter'",
        "Generate a 2-to-1 multiplexer with enable signal, name it 'mux_2to1",
        "Create a D flip-flop with asynchronous reset, name the module 'dff'",
    ]

    if user_input == "n":
        user_request = input("Enter your request:\n")
        test_requests = [user_request]

    for request in test_requests:
        print(f"\nRequest: {request}")
        print("=" * 60)

        result = generator.generate(
            request,
            output_dir="sv_output",
            recursion_limit=200,
            max_retries=10,
            max_regenerations=10,
        )

        print(f"Success: {result['success']}")
        print(f"Module directory: {result['module_dir'] or 'Not set'}")
        print(f"Cleanup performed: {result['cleanup_performed']}")
        print(
            f"Regeneration confirmed: {result['user_regenerate_confirmed']}"
        )
        print(f"Retry count: {result['retry_count']}")
        print(f"Regeneration count: {result['regeneration_count']}")

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
