"""Request detection utilities for API optimizations.

Detects quota checks, title generation, prefix detection, suggestion mode,
and filepath extraction requests to enable fast-path responses.
"""

from core.anthropic import extract_text_from_content

from .models.anthropic import MessagesRequest


def is_quota_check_request(request_data: MessagesRequest) -> bool:
    """Check if this is a quota probe request.

    Quota checks are typically simple requests with max_tokens=1
    and a single message containing the word "quota".
    """
    if (
        request_data.max_tokens == 1
        and len(request_data.messages) == 1
        and request_data.messages[0].role == "user"
    ):
        text = extract_text_from_content(request_data.messages[0].content)
        if "quota" in text.lower():
            return True
    return False


def is_title_generation_request(request_data: MessagesRequest) -> bool:
    """Check if this is a conversation title generation request.

    Title generation requests are detected by a system prompt containing
    title extraction instructions, no tools, and a single user message.

    Matches Claude Code session title prompts (sentence-case title, JSON
    \"title\" field, etc.).
    """
    if not request_data.system or request_data.tools:
        return False
    system_text = extract_text_from_content(request_data.system).lower()
    if "title" not in system_text:
        return False
    return "sentence-case title" in system_text or (
        "return json" in system_text
        and "field" in system_text
        and ("coding session" in system_text or "this session" in system_text)
    )


def is_prefix_detection_request(request_data: MessagesRequest) -> tuple[bool, str]:
    """Check if this is a fast prefix detection request.

    Prefix detection requests contain a policy_spec block and
    a Command: section for extracting shell command prefixes.

    Returns:
        Tuple of (is_prefix_request, command_string)
    """
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, ""

    content = extract_text_from_content(request_data.messages[0].content)

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except TypeError:
            return False, ""

    return False, ""


def is_safety_classifier_request(request_data: MessagesRequest) -> bool:
    """Check if this is a Claude Code auto-mode safety-classifier request.

    Auto mode sends a dedicated request asking a classifier to decide whether a
    tool call is safe to auto-approve. Claude Code wraps the action under review
    in a ``<transcript>`` block and parses the verdict from a literal
    ``<block>yes</block>`` / ``<block>no</block>`` tag in the response. These
    requests carry no tools.

    Reasoning ("thinking") models can answer in a native channel/JSON format
    (e.g. ``{"block":"no"}``) instead of that exact XML tag, which the classifier
    parser cannot read and reports as the model being temporarily unavailable.
    Detecting the request lets the proxy keep the model's output in the
    plain-text format the classifier expects.
    """
    if request_data.tools:
        return False

    system_text = (
        extract_text_from_content(request_data.system) if request_data.system else ""
    )
    messages_text = "".join(
        extract_text_from_content(message.content) for message in request_data.messages
    )
    combined = f"{system_text}\n{messages_text}"
    has_verdict_instruction = "yes</block>" in combined or "no</block>" in combined
    return "<transcript>" in combined and has_verdict_instruction


def is_suggestion_mode_request(request_data: MessagesRequest) -> bool:
    """Check if this is a suggestion mode request.

    Suggestion mode requests contain "[SUGGESTION MODE:" in the user's message,
    used for auto-suggesting what the user might type next.
    """
    for msg in request_data.messages:
        if msg.role == "user":
            text = extract_text_from_content(msg.content)
            if "[SUGGESTION MODE:" in text:
                return True
    return False


def is_filepath_extraction_request(
    request_data: MessagesRequest,
) -> tuple[bool, str, str]:
    """Check if this is a filepath extraction request.

    Filepath extraction requests have a single user message with
    "Command:" and "Output:" sections, asking to extract file paths
    from command output.

    Returns:
        Tuple of (is_filepath_request, command, output)
    """
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, "", ""
    if request_data.tools:
        return False, "", ""

    content = extract_text_from_content(request_data.messages[0].content)

    if "Command:" not in content or "Output:" not in content:
        return False, "", ""

    # Match if user content OR system block indicates filepath extraction
    user_has_filepaths = (
        "filepaths" in content.lower() or "<filepaths>" in content.lower()
    )
    system_text = (
        extract_text_from_content(request_data.system) if request_data.system else ""
    )
    system_has_extract = (
        "extract any file paths" in system_text.lower()
        or "file paths that this command" in system_text.lower()
    )
    if not user_has_filepaths and not system_has_extract:
        return False, "", ""

    cmd_start = content.find("Command:") + len("Command:")
    output_marker = content.find("Output:", cmd_start)
    if output_marker == -1:
        return False, "", ""

    command = content[cmd_start:output_marker].strip()
    output = content[output_marker + len("Output:") :].strip()

    for marker in ["<", "\n\n"]:
        if marker in output:
            output = output.split(marker)[0].strip()

    return True, command, output
