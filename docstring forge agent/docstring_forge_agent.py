from pathlib import Path
from typing import Annotated, List, Optional

from docstring_forge_handlers import DocstringForgeHandlers
from docstring_forge_tools import find_python_files_tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# Available LLM models for selection
LLM_MODELS = [
    "groq:llama-3.3-70b-versatile",
    "groq:deepseek-r1-distill-llama-70b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
]

# Global LLM instructions for docstring operations
LLM_INSTRUCTIONS = """
You are an expert Python docstring generator.
- Improve or add docstrings in the provided Python code, following Google
docstring conventions without the example section.
- Keep maximum 79 chars per line.
- Remove trailing whitespaces from docstrings.
- Add docstrings to all functions and classes that lack them.
- Do not modify other parts of the code or refactor it.

**Requirements**:
1. Clear, concise descriptions.
2. Document all parameters and return values.
3. Use consistent formatting.
4. Follow Google docstring style without examples.
5. Ensure docstrings are added for undocumented functions/classes.

**Input Format**:
- Current docstrings found: {docstrings_info}
- Original code:
```python
{original_code}
```

**Output Format**:
- Return the complete updated Python code with improved/added docstrings.
- Wrap the output in a single ```python code block.
- Do not include other code blocks or markers.
"""


class AgentState(TypedDict):
    """State for docstring and comment processing workflow.

    Attributes:
        file_path: Path to the Python file to process.
        original_code: Original Python code content.
        processed_code: Code after docstring/comment processing.
        action: Action to perform ('remove' or 'update').
        docstring_info: List of dictionaries with docstring details.
        messages: List of chat messages for LLM interaction.
        error: Error message if processing fails, None otherwise.
        output_dir: Directory to save processed files.
        saved_file: Path to the saved processed file.
        selected_model: The selected LLM model for docstring updates.
    """

    file_path: str
    original_code: str
    processed_code: str
    action: str
    docstring_info: List[dict]
    messages: Annotated[list, add_messages]
    error: Optional[str]
    output_dir: str
    saved_file: str
    selected_model: str


class DocstringForge:
    """Orchestrates the docstring processing workflow using LangGraph.

    Manages the workflow for finding, loading, analyzing, processing, and
    saving Python files with updated or removed docstrings and comments.

    Attributes:
        llm: Initialized language model for docstring updates.
        selected_model: Selected LLM model identifier.
        sv_prompt: Prompt template for LLM docstring generation.
        graph: Compiled LangGraph workflow for processing.
        handler: Instance of DocstringForgeHandlers for node operations.
    """

    def __init__(self, model: str = None):
        """Initialize the DocstringForge with workflow.

        Args:
            model: LLM model identifier (optional, set when needed).
        """
        self.llm = None
        self.selected_model = model
        self.sv_prompt = LLM_INSTRUCTIONS
        self.handler = DocstringForgeHandlers(None, self.sv_prompt)
        self.graph = self.create_workflow()

    def initialize_llm(self, model: str):
        """Initialize LLM with selected model.

        Args:
            model: LLM model identifier to initialize.
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=12000)
        self.selected_model = model
        self.handler.llm = self.llm
        # Initialize the prompt template now that LLM is available
        self.handler.sv_prompt = ChatPromptTemplate.from_messages([
            ("system", self.sv_prompt),
            (
                "user",
                "{docstrings_info}\n\nOriginal code:"
                "\n```python\n{original_code}\n```",
            ),
        ])

    def create_workflow(self) -> StateGraph:
        """Create and compile the LangGraph workflow for docstring processing.

        Returns:
            StateGraph: Compiled workflow for processing Python files.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("load", self.handler.load_file)
        workflow.add_node("analyze", self.handler.analyze_docstrings)
        workflow.add_node("process", self.handler.process_docstrings)
        workflow.add_node("llm", self.handler.llm_process)
        workflow.add_node("save", self.handler.save_result)
        workflow.add_edge(START, "load")
        workflow.add_edge("load", "analyze")
        workflow.add_edge("analyze", "process")
        workflow.add_conditional_edges(
            "process",
            self.handler.should_use_llm,
            {
                "use_llm": "llm",
                "skip_llm": "save",
            },
        )
        workflow.add_edge("llm", "save")
        workflow.add_edge("save", END)
        return workflow.compile()

    def select_model(self) -> str:
        """Prompt user to select an LLM model from available options.

        Returns:
            str: Selected LLM model identifier.
        """
        print("Available LLM Models:")
        for model_index, model in enumerate(LLM_MODELS, 1):
            print(f"{model_index}. {model}")
        while True:
            choice = input(
                f"\nSelect a model (1-{len(LLM_MODELS)}): "
            ).strip()
            try:
                selected_model_index = int(choice) - 1
                if 0 <= selected_model_index < len(LLM_MODELS):
                    return LLM_MODELS[selected_model_index]
                print(f"Select a number between 1 and {len(LLM_MODELS)}.")
            except ValueError:
                print("Invalid input.")

    def process_file(
        self, action: str, file_path: Path, output_dir: str
    ) -> dict:
        """Process a single Python file with the specified action.

        Args:
            action: Action to perform ('remove' or 'update').
            file_path: Path to the Python file to process.
            output_dir: Directory to save the processed file.

        Returns:
            dict:
                Result containing success status, processed code,
                and messages.
        """
        rel_path = file_path.relative_to(Path.cwd())
        print(f"🔧 Processing: {rel_path}")
        print(f"📝 Action: {action}")
        print(f"📂 Output: {output_dir}")
        print("-" * 50)

        initial_state = {
            "file_path": str(file_path),
            "action": action,
            "original_code": "",
            "processed_code": "",
            "docstring_info": [],
            "messages": [],
            "error": None,
            "output_dir": output_dir,
            "saved_file": "",
            "selected_model": self.selected_model,
        }

        try:
            result = self.graph.invoke(initial_state)
            success = not bool(result.get("error"))
            if success:
                print(
                    f"✅ {'Removed' if action == 'remove' else 'Updated'} "
                    f"{len(result['docstring_info'])} docstrings"
                )
            else:
                print(f"❌ Error: {result['error']}")
            print("✨ Done!")
            return {
                "success": success,
                "processed_code": result["processed_code"],
                "messages": result["messages"],
                "error": result["error"],
                "saved_file": result["saved_file"],
                "docstring_info": result["docstring_info"],
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "success": False,
                "processed_code": "",
                "messages": [],
                "error": str(e),
                "saved_file": "",
                "docstring_info": [],
            }

    def interactive_mode(self):
        """Run the docstring forge in interactive mode."""
        current_dir = Path.cwd()
        print("🔥 Docstring Forge - Interactive")
        print(f"📂 Scanning: {current_dir}")
        print("=" * 60)

        while True:
            try:
                result = find_python_files_tool.invoke({
                    "directory": str(current_dir)
                })
                if result["error"]:
                    print(f"❌ Error: {result['error']}")
                    return

                python_files = [Path(f) for f in result["python_files"]]
                if not python_files:
                    print("❌ No Python files found.")
                    return

                print(f"✅ Found {len(python_files)} Python files")
                self.handler.display_files_menu(python_files)
                action, selected_file, output_dir = (
                    self.handler.get_user_choice(python_files)
                )

                # Initialize LLM only for update action
                if action == "update" and self.llm is None:
                    selected_model = self.select_model()
                    self.initialize_llm(selected_model)
                    print(f"\n🤖 Selected LLM Model: {selected_model}")

                max_repeats = 5
                repeat_count = 0
                while repeat_count < max_repeats:
                    self.process_file(action, selected_file, output_dir)
                    repeat_count += 1
                    if repeat_count < max_repeats:
                        print(f"🔄 Repeated {repeat_count} of {max_repeats}")
                        choice = (
                            input(
                                f"🔄 Repeat on {selected_file.name}? (y/n): "
                            )
                            .lower()
                            .strip()
                        )
                        if choice not in ["y", "yes"]:
                            break
                    else:
                        print(f"🔄 Max repeats ({max_repeats}) reached")

                choice = (
                    input("\n🔄 Process another file? (y/n): ")
                    .lower()
                    .strip()
                )
                if choice not in ["y", "yes"]:
                    print("👋 Goodbye!")
                    return
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    forge = DocstringForge()
    print("Graph structure:")
    print(forge.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    forge.interactive_mode()
