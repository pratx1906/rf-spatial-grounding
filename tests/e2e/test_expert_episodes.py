import statistics


def test_expert_solves_m1_tasks_end_to_end(env, tasks):
    from tasks.expert import solve

    by_area = {}
    for t in tasks:
        env.reset(t)
        solve(env, t)
        assert env.done and env.end_reason == "submitted", (t.task_id, env.end_reason, env.verifier_result)
        assert env.turn <= env.cfg.max_turns
        by_area.setdefault(t.template, []).append(env.answer_score)
    for tmpl, scores in by_area.items():
        assert statistics.mean(scores) >= 0.8, (tmpl, scores)
