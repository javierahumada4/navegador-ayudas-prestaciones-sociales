"""A minimal trace viewer: send a task to the running API and see every step as a card.

    uv run python -m agent.ui            # expects the API at http://localhost:8000

Same idea as the coloured trace of the class notebook: purple for thoughts, orange for
actions, green for observations. It is your debugging tool and your demo.
"""

from __future__ import annotations

import html
import json
import os

import httpx

API_URL = os.environ.get("ARCA_API_URL", "http://localhost:8000")
COLORS = {
    "thought": ("#ede9fe", "#4c1d95"),
    "action": ("#ffedd5", "#9a3412"),
    "observation": ("#dcfce7", "#166534"),
    "final": ("#ccfbf1", "#115e59"),
}


def card(title: str, body: str, kind: str) -> str:
    bg, fg = COLORS[kind]
    return (
        f'<div style="background:{bg};color:{fg};border-left:6px solid {fg};padding:10px 14px;'
        f'margin:8px 0;border-radius:8px"><b>{html.escape(title)}</b>'
        f'<pre style="white-space:pre-wrap;margin-top:8px;color:{fg}">'
        f"{html.escape(body)}</pre></div>"
    )


def render_trace(response: dict) -> str:
    blocks = []
    for i, step in enumerate(response.get("steps", []), start=1):
        blocks.append(card(f"STEP {i} · Thought", step["thought"], "thought"))
        if step.get("action"):
            blocks.append(
                card("Action", json.dumps(step["action"], ensure_ascii=False, indent=2), "action")
            )
        if step.get("observation") is not None:
            blocks.append(card("Observation", str(step["observation"]), "observation"))
    blocks.append(card("FINAL ANSWER", response.get("final_answer", ""), "final"))
    return "\n".join(blocks)


def run(task: str, brain: str, max_steps: int, session_id: str = "") -> tuple[str, str]:
    payload = {"task": task, "brain": brain, "max_steps": int(max_steps)}
    if session_id.strip():
        payload["session_id"] = session_id.strip()  # same id = multi-turn conversation
    reply = httpx.post(f"{API_URL}/agent", json=payload, timeout=600)
    if reply.status_code != 200:
        return f"API error {reply.status_code}: {reply.text}", ""
    body = reply.json()
    header = (
        f"<p><b>{html.escape(body.get('model', ''))}</b> · {body['n_steps']} pasos · "
        f"{body.get('tokens_used', 0)} tokens · "
        f"{'terminó' if body.get('succeeded', True) else 'NO terminó'}</p>"
    )
    return body["final_answer"], header + render_trace(body)


def main() -> None:
    import gradio as gr

    with gr.Blocks(title="ARCA · agent trace") as demo:
        gr.Markdown("# ARCA · ReAct trace viewer")
        task = gr.Textbox(label="Task", lines=3)
        with gr.Row():
            brain = gr.Dropdown(["base", "rlm", "thinking"], value="rlm", label="Brain")
            max_steps = gr.Slider(1, 30, value=10, step=1, label="Max steps")
            session = gr.Textbox(label="Session id (optional, multi-turn)", value="")
        button = gr.Button("Run", variant="primary")
        answer = gr.Textbox(label="Final answer", lines=4, interactive=False)
        trace = gr.HTML(label="Trace")
        button.click(run, inputs=[task, brain, max_steps, session], outputs=[answer, trace])
    demo.launch()


if __name__ == "__main__":
    main()
