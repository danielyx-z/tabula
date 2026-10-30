from pathlib import Path

# Fix audio2tab.py - prompt_tabs should always be 0
p = Path("scripts/audio2tab.py")
s = p.read_text()

old = """            prompt_tabs = 1 if chunk.meta.get("has_context") else 0"""
new = """            # prompt_tabs=0: the context TAB in the decoder prompt is counted
            # in tab_count, which correctly shifts the index to the first
            # target note's pitch.  prompt_tabs>0 would under-index."""
            prompt_tabs = 0"""
assert s.count(old) == 1, "audio2tab.py: prompt_tabs line not found"
s = s.replace(old, new)

p.write_text(s)
print("scripts/audio2tab.py: prompt_tabs fixed to 0")

# Fix cli.py - same issue
p = Path("src/fretwire/cli.py")
s = p.read_text()

old = """            prompt_tabs = 1 if chunk.meta.get("has_context") else 0"""
new = """            # prompt_tabs=0: the context TAB in the decoder prompt is counted
            # in tab_count, which correctly shifts the index to the first
            # target note's pitch.  prompt_tabs>0 would under-index."""
            prompt_tabs = 0"""
assert s.count(old) == 1, "cli.py: prompt_tabs line not found"
s = s.replace(old, new)

p.write_text(s)
print("src/fretwire/cli.py: prompt_tabs fixed to 0")

PYEOF
