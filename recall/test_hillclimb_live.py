"""Oracle and null sessions through the grader, before any paid run.

Run: python3 -m unittest discover -s recall -p 'test_*.py'
"""

from __future__ import annotations

import json
import unittest

import check_live as live
import hillclimb_live as hc

MEMBERS = ["llm-application-engineering", "ml-project-lifecycle"]


def session(*blocks: dict, model: str = "claude-opus-5-5", subtype: str = "success") -> list[str]:
    """A stream-json transcript with one assistant turn holding `blocks`."""
    return [
        json.dumps({"type": "system", "subtype": "init", "model": model, "skills": [], "plugins": []}),
        json.dumps({
            "type": "assistant",
            "parent_tool_use_id": None,
            "message": {"model": model, "content": [*blocks, {"type": "text", "text": "done"}]},
        }),
        json.dumps({"type": "result", "subtype": subtype, "usage": {}, "total_cost_usd": 0, "num_turns": 1}),
    ]


def router_then_read(member: str) -> list[str]:
    return session(
        {"type": "tool_use", "name": "Skill", "input": {"skill": "ai-ml:ai-ml"}},
        {"type": "tool_use", "name": "Read",
         "input": {"file_path": f"/x/skills/ai-ml/members/{member}/SKILL.md"}},
    )


def graded(lines: list[str], expected: str, router: str | None = "ai-ml") -> dict[str, float]:
    fired, member, _ = live.observe(lines, router, MEMBERS, {})
    return hc.grade(expected, member, fired)


class OracleAndNull(unittest.TestCase):
    def test_oracle_scores_every_positive_correct(self):
        for m in MEMBERS:
            self.assertEqual(graded(router_then_read(m), m), {"correct": 1.0, "fired": 1.0})
        self.assertEqual(graded(router_then_read(MEMBERS[0]), "any"), {"correct": 1.0, "fired": 1.0})

    def test_flat_skill_call_counts_as_reached(self):
        lines = session({"type": "tool_use", "name": "Skill", "input": {"skill": "ml-project-lifecycle"}})
        self.assertEqual(graded(lines, "ml-project-lifecycle", router=None), {"correct": 1.0, "fired": 1.0})

    def test_null_scores_positives_wrong_and_negatives_right(self):
        silent = session()
        self.assertEqual(graded(silent, MEMBERS[0]), {"correct": 0.0, "fired": 0.0})
        self.assertEqual(graded(silent, "any"), {"correct": 0.0, "fired": 0.0})
        self.assertEqual(graded(silent, "none"), {"correct": 1.0, "fired": 0.0})

    def test_wrong_member_fires_but_is_not_correct(self):
        lines = router_then_read("ml-project-lifecycle")
        self.assertEqual(graded(lines, "llm-application-engineering"), {"correct": 0.0, "fired": 1.0})

    def test_router_alone_on_a_negative_is_a_false_fire(self):
        lines = session({"type": "tool_use", "name": "Skill", "input": {"skill": "ai-ml"}})
        self.assertEqual(graded(lines, "none"), {"correct": 0.0, "fired": 1.0})

    def test_served_model_ignores_subagent_turns(self):
        lines = session(model="claude-opus-5-5") + [json.dumps({
            "type": "assistant", "parent_tool_use_id": "t1",
            "message": {"model": "claude-haiku-5-5", "content": []},
        })]
        self.assertEqual(hc.served_models(hc.events(lines)), {"claude-opus-5-5"})

    def test_transcript_puts_prompt_after_system_and_keeps_tool_calls(self):
        turns = hc.transcript(hc.events(router_then_read(MEMBERS[0])), "the prompt")
        self.assertEqual([t["role"] for t in turns], ["system", "user", "tool_call", "tool_call", "assistant"])
        self.assertEqual(turns[1]["content"], "the prompt")


if __name__ == "__main__":
    unittest.main()
