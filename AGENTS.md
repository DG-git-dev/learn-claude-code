# Project Learning Workflow

When the user explicitly enters a new `sXX` lesson:

1. Read that lesson's `code.py` and README for context.
2. Modify only that lesson's `code.py` to use the OpenAI-compatible Responses API.
3. Preserve the lesson's original teaching behavior and architecture.
4. Do not modify README files unless the user explicitly asks.
5. Run a syntax check, focused checks for the lesson's new mechanism, and a safe end-to-end API test.
6. After all checks pass, commit only that lesson's `code.py` in a dedicated commit. Do not push unless the user asks.
7. Use the `show-me` skill when explaining the newly entered lesson. For this lesson-entry explanation, use a Mermaid flowchart that maps the lesson's actual `code.py` functions, tool sets, loops, branches, and data flow rather than showing only a high-level concept. Do not create an HTML visualization unless requested.
8. Add a concise docstring to every function in the lesson's `code.py`, plus inline comments for important or non-obvious lines. Avoid comments that merely restate the code.
9. Center the explanation on what this `sXX` lesson adds. Explicitly show how the new mechanism connects to, reuses, or changes the structures introduced in earlier lessons; avoid spending equal attention on unchanged inherited code.

These `show-me` constraints apply only to the new-lesson workflow above. In all other scenarios, use the `show-me` skill for its original general-purpose visual explanation behavior and choose the visual format that best fits the user's request.

Never stage generated practice files or unrelated working-tree changes with a lesson commit.
