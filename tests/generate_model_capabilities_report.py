import html
import json
from pathlib import Path


def generate_markdown_report(
    no_hacks_path: str = "test_output/hacks_disabled",
    hacks_path: str = "test_output/hacks_enabled",
    output_path: str = "model_capabilities.md",
):
    """
    Generates a Markdown report comparing model capabilities with and without hacks.

    It reads two JSON files containing test results, one with hacks enabled and one
    without. It then generates a Markdown table that summarizes the capabilities
    of each model, highlighting any improvements gained from using the hacks.

    Args:
        no_hacks_path: Path to the directory with results when hacks are disabled.
        hacks_path: Path to the directory with results when hacks are enabled.
        output_path: Path to write the output Markdown file.
    """
    no_hacks_dir = Path(no_hacks_path)
    hacks_dir = Path(hacks_path)

    def load_results(directory: Path) -> dict:
        results = {}
        if not directory.exists():
            print(f"WARNING: '{directory}' not found. Skipping.")
            return results
        for file in directory.glob("*.json"):
            data = json.loads(file.read_text(encoding="utf-8"))
            results.update(data)
        return results

    no_hacks_data = load_results(no_hacks_dir)
    hacks_data = load_results(hacks_dir)

    capabilities = [
        "can_json",
        "can_pydantic",
        "can_tool_call",
        "can_think",
        "content_no_thinking",
    ]

    all_models = list(set(no_hacks_data.keys()) | set(hacks_data.keys()))

    def calculate_score(model_name):
        hacks_results = hacks_data.get(model_name, {})
        score = sum(1 for cap in capabilities if hacks_results.get(cap, [False])[0])
        return score

    all_models.sort(key=lambda m: (-calculate_score(m), m))
    header_map = {
        "can_think": "Thinking output",
        "content_no_thinking": "No thinking in Content",
        "can_json": "JSON Format",
        "can_pydantic": "Pydantic Format",
        "can_tool_call": "Tool Calls",
    }

    md_lines = [
        "# Model Capability Report",
        "",
        "This report compares model capabilities with and without `ollama-think`'s compatibility hacks.",
        "A `❌` &rarr; `✅` indicates that the hack fixed a previously failing capability.",
        "A `❗` indicates invalid JSON, on one test without specific encouragement.",
        "",
    ]

    header = "| Model | " + " | ".join(header_map[c] for c in capabilities) + " |"
    md_lines.append(header)
    separator = "|:---| " + " | ".join([":---"] * len(capabilities)) + " |"
    md_lines.append(separator)

    def format_icon(res):
        if not res:
            return ""
        ok = res[0]
        desc = html.escape(res[1][:50].replace("\n", " ").replace("|", ""))
        if ok:
            return "✅"
        if "Invalid JSON" in res[1] or "Expecting" in res[1]:
            return """[❗](## "Invalid JSON")"""
        return f'''[❌](## "{desc}")'''

    def format_cell(no_hacks_res, hacks_res):
        no_hacks_icon = format_icon(no_hacks_res)
        hacks_icon = format_icon(hacks_res)

        if no_hacks_res and hacks_res and no_hacks_res[0] != hacks_res[0]:
            return f"{no_hacks_icon} &rarr; {hacks_icon}"
        return hacks_icon

    for model in all_models:
        row = [f"`{model}`"]
        no_hacks_results = no_hacks_data.get(model, {})
        hacks_results = hacks_data.get(model, {})

        for cap in capabilities:
            no_hacks_res = no_hacks_results.get(cap)
            hacks_res = hacks_results.get(cap)
            row.append(format_cell(no_hacks_res, hacks_res))

        md_lines.append("| " + " | ".join(row) + " |")

    output_file = Path(output_path)
    output_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Markdown report generated at: {output_path}")


if __name__ == "__main__":
    generate_markdown_report()
