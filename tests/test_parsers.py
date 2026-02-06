from providers.nvidia_nim.utils.think_parser import ThinkTagParser, ContentType
from providers.nvidia_nim.utils.heuristic_tool_parser import HeuristicToolParser


def test_think_tag_parser_basic():
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello <think>reasoning</think> world"))

    assert len(chunks) == 3
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello "
    assert chunks[1].type == ContentType.THINKING
    assert chunks[1].content == "reasoning"
    assert chunks[2].type == ContentType.TEXT
    assert chunks[2].content == " world"


def test_think_tag_parser_streaming():
    parser = ThinkTagParser()

    # Partial tag
    chunks = list(parser.feed("Hello <thi"))
    assert len(chunks) == 1
    assert chunks[0].content == "Hello "

    # Complete tag
    chunks = list(parser.feed("nk>reasoning</think>"))
    assert len(chunks) == 1
    assert chunks[0].type == ContentType.THINKING
    assert chunks[0].content == "reasoning"


def test_heuristic_tool_parser_basic():
    parser = HeuristicToolParser()
    text = "Let's call a tool. ● <function=Grep><parameter=pattern>hello</parameter><parameter=path>.</parameter>"
    filtered, tools_initial = parser.feed(text)
    tools_final = parser.flush()
    tools = tools_initial + tools_final

    assert "Let's call a tool." in filtered
    assert len(tools) == 1
    assert tools[0]["name"] == "Grep"
    assert tools[0]["input"] == {"pattern": "hello", "path": "."}


def test_heuristic_tool_parser_streaming():
    parser = HeuristicToolParser()

    # Feed part 1
    filtered1, tools1 = parser.feed("● <function=Write>")
    assert tools1 == []

    # Feed part 2
    filtered2, tools2 = parser.feed("<parameter=path>test.txt</parameter>")
    assert tools2 == []

    # Feed part 3 (triggering flush or completion)
    filtered3, tools3 = parser.feed("\nDone.")
    assert len(tools3) == 1
    assert tools3[0]["name"] == "Write"
    assert tools3[0]["input"] == {"path": "test.txt"}
    assert "Done." in filtered3


def test_heuristic_tool_parser_flush():
    parser = HeuristicToolParser()
    parser.feed("● <function=Bash><parameter=command>ls -la")
    tools = parser.flush()

    assert len(tools) == 1
    assert tools[0]["name"] == "Bash"
    assert tools[0]["input"] == {"command": "ls -la"}


def test_interleaved_thinking_and_tools():
    parser_think = ThinkTagParser()
    parser_tool = HeuristicToolParser()

    text = "<think>I need to search for a file.</think> ● <function=Grep><parameter=pattern>test</parameter>"

    # 1. Parse thinking
    chunks = list(parser_think.feed(text))
    thinking = [c for c in chunks if c.type == ContentType.THINKING]
    text_remaining = "".join([c.content for c in chunks if c.type == ContentType.TEXT])

    assert len(thinking) == 1
    assert thinking[0].content == "I need to search for a file."

    # 2. Parse tool from remaining text
    filtered, tools = parser_tool.feed(text_remaining)
    tools += parser_tool.flush()

    assert len(tools) == 1
    assert tools[0]["name"] == "Grep"
    assert tools[0]["input"] == {"pattern": "test"}


def test_partial_interleaved_streaming():
    parser_think = ThinkTagParser()
    parser_tool = HeuristicToolParser()

    # Chunk 1: Partial thinking (it emits since it's definitely not the start of <think>)
    chunks1 = list(parser_think.feed("<think>Part 1"))
    assert len(chunks1) == 1
    assert chunks1[0].type == ContentType.THINKING
    assert chunks1[0].content == "Part 1"

    # Chunk 2: Thinking ends, tool starts
    chunks2 = list(parser_think.feed(" ends</think> ● <func"))
    assert len(chunks2) == 2
    assert chunks2[0].type == ContentType.THINKING
    assert chunks2[0].content == " ends"

    text_rem = chunks2[1].content
    filtered, tools = parser_tool.feed(text_rem)
    assert tools == []

    # Chunk 3: Tool ends
    chunks3 = list(parser_think.feed("tion=Read><parameter=path>test.py</parameter>"))
    text_rem3 = "".join([c.content for c in chunks3])
    filtered3, tools3 = parser_tool.feed(text_rem3)
    tools3 += parser_tool.flush()

    assert len(tools3) == 1
    assert tools3[0]["name"] == "Read"
    assert tools3[0]["input"] == {"path": "test.py"}


# --- New Robustness Tests ---


def test_split_across_markers():
    # Split across the trigger chaaracter
    # "● <function=Test>"
    # Split at various points
    full_text = "● <function=Test><parameter=arg>val</parameter>"

    for i in range(len(full_text)):
        p = HeuristicToolParser()
        chunk1 = full_text[:i]
        chunk2 = full_text[i:]

        tools = []
        filtered, t = p.feed(chunk1)
        tools.extend(t)
        filtered2, t = p.feed(chunk2)
        tools.extend(t)
        tools.extend(p.flush())

        if len(tools) != 1:
            print(f"Failed split at index {i}: '{chunk1}' | '{chunk2}'")

        assert len(tools) == 1, f"Failed split at index {i}"
        assert tools[0]["name"] == "Test"
        assert tools[0]["input"] == {"arg": "val"}


def test_value_with_special_chars():
    parser = HeuristicToolParser()
    # Value with > inside
    text = "● <function=Test><parameter=arg>a > b</parameter>"
    _, tools = parser.feed(text)
    tools.extend(parser.flush())

    assert len(tools) == 1
    assert tools[0]["input"]["arg"] == "a > b"


def test_multiple_params_split():
    full_text = (
        "● <function=Test><parameter=p1>v1</parameter><parameter=p2>v2</parameter>"
    )

    for i in range(len(full_text)):
        p = HeuristicToolParser()
        tools = []
        _, t = p.feed(full_text[:i])
        tools.extend(t)
        _, t = p.feed(full_text[i:])
        tools.extend(t)
        tools.extend(p.flush())

        assert len(tools) == 1, f"Failed split at {i}"
        assert tools[0]["input"] == {"p1": "v1", "p2": "v2"}


def test_incomplete_tag_flush():
    p = HeuristicToolParser()
    p.feed("● <function=Recover><parameter=msg>hello")
    tools = p.flush()

    assert len(tools) == 1
    assert tools[0]["input"]["msg"] == "hello"


def test_garbage_interleaved():
    p = HeuristicToolParser()
    tools = []
    _, t = p.feed("Some text ")
    tools.extend(t)
    _, t = p.feed("● <function=T1><parameter=x>1</parameter>")
    tools.extend(t)
    _, t = p.feed(" more text ")
    tools.extend(t)
    _, t = p.feed("● <function=T2><parameter=y>2</parameter>")
    tools.extend(t)
    tools.extend(p.flush())

    assert len(tools) == 2
    assert tools[0]["name"] == "T1"
    assert tools[1]["name"] == "T2"


def test_text_between_params_lost():
    p = HeuristicToolParser()
    # " text1 " is between function end and first param
    # " text2 " is between params
    text = "● <function=F> text1 <parameter=a>1</parameter> text2 <parameter=b>2</parameter>"
    filtered, tools = p.feed(text)
    tools.extend(p.flush())

    # Check if "text1" and "text2" are preserved in filtered output
    assert "text1" in filtered
    assert "text2" in filtered
    assert tools[0]["input"] == {"a": "1", "b": "2"}


# --- Orphan </think> Tag Tests (Step Fun AI compatibility) ---


def test_orphan_close_tag_stripped():
    """Orphan </think> without opening tag should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello </think> world"))

    # Should get one text chunk with orphan tag stripped
    assert len(chunks) == 2
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello "
    assert chunks[1].type == ContentType.TEXT
    assert chunks[1].content == " world"


def test_orphan_close_tag_at_start():
    """Orphan </think> at start should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("</think>Hello world"))

    assert len(chunks) == 1
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello world"


def test_orphan_close_tag_at_end():
    """Orphan </think> at end should be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("Hello world</think>"))

    assert len(chunks) == 1
    assert chunks[0].type == ContentType.TEXT
    assert chunks[0].content == "Hello world"


def test_multiple_orphan_close_tags():
    """Multiple orphan </think> tags should all be stripped."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("a</think>b</think>c"))

    text = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    assert text == "abc"
    assert "</think>" not in text


def test_orphan_close_tag_streaming():
    """Orphan </think> split across chunks should be stripped."""
    parser = ThinkTagParser()

    # Feed partial orphan tag
    chunks1 = list(parser.feed("Hello </thi"))
    assert len(chunks1) == 1
    assert chunks1[0].content == "Hello "

    # Complete the orphan tag
    chunks2 = list(parser.feed("nk> world"))
    assert len(chunks2) == 1
    assert chunks2[0].type == ContentType.TEXT
    assert chunks2[0].content == " world"


def test_orphan_close_with_valid_think_pair():
    """Orphan </think> followed by valid <think>...</think> pair."""
    parser = ThinkTagParser()
    chunks = list(parser.feed("a</think>b<think>thinking</think>c"))

    types = [c.type for c in chunks]
    contents = [c.content for c in chunks]

    assert ContentType.TEXT in types
    assert ContentType.THINKING in types
    # Text should be "ab" and "c", thinking should be "thinking"
    text_content = "".join(c.content for c in chunks if c.type == ContentType.TEXT)
    think_content = "".join(c.content for c in chunks if c.type == ContentType.THINKING)
    assert text_content == "abc"
    assert think_content == "thinking"
