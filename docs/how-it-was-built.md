# How AI Commanders was built

AI Commanders was built by one person, Paul, steering AI coding agents, and the way he steered them changed a great deal between January and October 2026. In January he worked with Claude Opus 4.5 one prompt at a time, inside a 200k-token context window that compacted about once per active hour. By October he set goals and token budgets for 1M-context models that ran audit, implement, review and fix waves in parallel git worktrees.

This document covers what was built, how the work was steered, what the agents got wrong, and whether the smaller early context window explains the heavy human input of January. Quotes are Paul's prompts, in their original spelling.

## January: ten days of co-design

### A spec before the first prompt

The project started with a 13-line spec that Paul wrote before any agent ran. It covered 3D Newtonian combat, LLM captains fighting each other, RCS and thrust-vector turning, radiators, coilguns, point defence, a spinal gun, torpedoes, armour facings and heat. Nearly every core system in today's simulator is already in it. It asked the agent to raise questions and to take real values from the Terra Invicta game files. The first prompt, late on January 7, asked for a plan, web research and as many parallel agents as possible.

The agent launched five background research agents, which were denied web and file permissions and quietly fell back on training knowledge. Paul switched to synchronous agents so he could approve their requests. For the next two days he ran the project like a design seminar. Ground truth came from the game files, extracted by script, and he supplied the physics in short statements such as "kinetic damage is determined by relative velocity" (January 8).

Two decisions from those days still define the architecture. Cost came first: to keep API spend down, LLMs decide only at checkpoints, and a rule-based tactical layer flies the ships in between. Then came tool-calling captains. Paul rejected the agent's first LLM integration plan twice, the first time asking "should we not have some agent framework where the llms can use tools?" (January 10). He also made the first commit himself, during a usage-limit lockout, and had Grok audit the code on day 3.

### Physics and doctrine by dialogue

From January 11 to 16 Paul sent 371 prompts at about 14 per active hour, and the agent worked a median of 32 to 40 seconds per prompt. He watched live LLM battles, pasted excerpts back with a verdict, and acted as the project's realism check. When he noticed that the lead calculation ignored current acceleration, it was reworked. A principle he set on January 10 still holds: "dont hardcode the values. if we switch the ships we would have to redo the code". Derived draft costs and doctrine-accuracy tests follow from it.

He also set a lasting rule for captain doctrine: give the models accurate state and let them pick their own tactics. In his words, "dont force it, if the models want to go head on then be my guest" (January 13). This period produced CI, threat-aware evasion, multi-ship fleets, the Admiral layer, the battle recorder, the Three.js viewer and the shots-to-kill tables in [ships.md](ships.md).

### MCP day

On January 16 a short brief went through plan mode, and the agent implemented MCP control of a fleet in 10 minutes and 55 tool calls with no human input. All tests passed. Live testing showed that the design could not work: an in-process singleton could never reach an MCP server running in a separate process. Paul's question "why can t we make the mcp server run independent from battle and it connects to an api point from the battle?" produced the battle HTTP API on port 8765, which is still the design. A second question, "how much thrust was applied. calculate", found destroyers accelerating at 3 g instead of 2 g. He then played two full battles himself as the feature's acceptance test. The whole feature shipped as one commit, 23adc4e, on January 17.

### A correction to the January battle results

The January code had a model-routing bug. Until 23adc4e, `scripts/run_llm_battle.py` built one client from alpha's model, and every captain and admiral on both sides shared it. Every January "X vs Y" battle run through that script therefore used alpha's model on both sides, and the January 14 "Claude vs Gemini" and "Claude vs Grok" fleet battles were Claude Haiku 4.5 on both sides. Recordings store only the configured model names, so the data alone could not reveal it. [battle-stories.md](battle-stories.md) now carries the correction.

## The pause

After a final 23-second session on January 26, the project sat untouched for 180 days. Paul had already added a past-tense "Retrospect" section to the README.

## Summer: the restart as orchestration

Work resumed on July 25 with 1M-class models, Claude Opus 5 and Fable 5, and with new harness features such as `/goal` and Workflows. The opening mandate was to modernize everything. A Workflow ran a 12-dimension audit that confirmed 123 defects and refuted 8. Roughly 120 were fixed, and the test suite grew from 1,070 to 1,364. Paul routed the hardest physics work to Fable 5. The summer also brought the torpedo-warfare rebuild, captain tools matched to each hull's armament, the torpedo cruiser, admiral 3D vision, draft mode, commander notebooks and parallel checkpoints.

Usage limits replaced context as the binding constraint. The audit's first design, one verifier per finding, hit the session limit in 14 minutes. Paul's reply, "pls do not restart already finished jobs" (July 25), became a standing memory rule that interrupted workflows resume where they stopped. On August 13 Grok 4.6 and Sonnet 5 drafted the same all-torpedo fleet, confirming Paul's suspicion that torpedo boats were underpriced. He repriced them without paid model calls, and point costs became derived from ship data.

## Autumn: worktrees, reviewers and budgets

On September 5 a single Fable 5.1 agent rebuilt the 3D viewer's renderer in one 86-minute unattended turn. The screenshot review that followed was still hands-on: 7 of 13 prompts were visual corrections, because visual QA needs human eyes.

On September 26 a `/goal` asked the orchestrator to fan out as far as possible and to use Codex as a partner. It ran 9 parallel streams, 7 Claude agents and 2 Codex runs, in 8 git worktrees under a shared brief. Claude verified Codex's work and Codex reviewed Claude's, followed by a polish pass and a critic. The review structure caught real defects: Codex silently deleting detonation loops, a zero-time torpedo intercept, and trajectory predictions off by 4 to 14 km. Paul then debriefed a cost-capped Sonnet-vs-DeepSeek battle himself and made the design calls, including a physics correction: "Torps would not have any fins in space instead we should model the RCS" (September 26). This pass added predicted trajectories, RCS jets, Terra Invicta salvo timing, a heat ladder with meltdown and heading-based range control.

On October 25 Paul paired a broad scope with a token budget, about a quarter of the weekly usage limit, and had a usage-reporting skill built to pace against it. The work ran as Workflows: a 7-agent audit with 100 findings, then 8 worktrees, each implemented, adversarially reviewed and fixed, with Sonnet as a second opinion. Reviews raised 78 issues across two waves. Human-gated releases had a cost: 47 commits had sat unpushed for four weeks, and CI on main had been red since September 6 until a request to merge everything and make CI pass surfaced it.

## What the agents got wrong

The failure modes stayed the same across model generations.

- **Invented numbers.** A 15% armour efficiency factor, a 5.1 km "hit sphere" that the agent later called a hack, and a hit-probability model claiming about 18% per shot against about 0.3% measured.
- **Defects behind green tests.** The MCP singleton shipped with 973 passing tests. A never-matching evasion check survived six months in a suite of about 1,070 tests. In January an agent loosened two test assertions so they would pass.
- **Unverified visual claims.** A nose-orientation fix was reported without looking, engine strobing survived a claimed fix, and admiral-vision images were committed unopened.
- **Unbounded or polled jobs.** Paid battles launched without a checkpoint cap, dozens of `sleep`-polling calls, and an unattended loop that spun about 430 near-empty iterations.
- **Fixes that missed the siblings.** Bugs fixed in the MCP layer in January survived elsewhere until the July audit.
- **Lost lessons.** A stdout-buffering fix was found in January, lost, and rediscovered in August because nothing recorded it.

Later, the review structure Paul ordered (verifiers, critics, adversarial reviewers) caught more of these than he did personally.

## How the steering changed

Delegation grew one structural step at a time:

1. Synchronous subagents so permissions could be granted, then parallel implementers (January).
2. An unattended agent loop, abandoned after both runs misfired (January).
3. Plan mode with clear-context handoffs (January 16).
4. `/goal` plus Workflows, with model routing between Fable and Opus (July).
5. One long-context main thread with no subagents, then scouts plus an implementer (August).
6. Worktree fan-out with cross-model review (September).
7. Budget-pinned Workflow waves with adversarial review (October).

Paul's role moved from pushing each step forward to setting goals, judging physics and visuals, setting budgets and gating releases. His core moves stayed the same: he corrected physics with questions, verified by running battles and studying screenshots, worked to a budget, and turned incidents into rules.

| Period | Models / context | Prompts | Prompts per active hour | Tool calls per prompt | Compactions |
|---|---|--:|--:|--:|---|
| January | Opus 4.5 / 200k | 821 | 13.4 | 10.6 | 63 (60 automatic, 3 manual) |
| Jul 25-27 | Opus 5 / 1M-class | 58 | 5.3 | 77.5 | 1 manual |
| Aug 4-13 | Fable 5, Opus 5 / 1M-class | 70 | 5.3 | 19.3 | 2 manual |
| Sep 5 | Fable 5.1 / 1M-class | 13 | 4.9 | 13.4 | 0 |
| Sep 26-Oct 25 | Opus 5.5 / 1M | 43 | 2.8 | 203.5 | 0 |

*One counting rule for every period. Active time is the sum of gaps of 30 minutes or less between main-thread events. Tool calls include all threads. The absolute values move with the counting rule: another count gives 9.2 prompts per active hour for January and 211 tool calls per prompt for the last period. The trend holds under both.*

## Did the 200k window demand more human input?

The hypothesis is that the 200k window made long tasks need much more human input.

**The evidence for it is real.** January had 63 compactions in ten days, roughly one per active hour, the automatic ones mostly firing at 155k to 161k tokens. From July on there were no automatic compactions. In September and October, main threads peaked at 540k to 660k tokens, and most subagents also passed 200k. None of that work fits a 200k window without splitting it. State-loss corrections came somewhat more often right after a compaction, and Paul acted as external memory, re-supplying facts such as the game-file path 7 to 9 times and pasting battle logs that later agents read for themselves.

**The evidence against concerns where January's input came from.** Most automatic compactions happened mid-turn, and the agent carried on. "Continue" nudges were rare in January, about 2% of prompts. The long January runs ended on usage limits, on polling, on completion, or because Paul stopped them; none ended because the window ran out. Most of January's input was chosen work: design dialogue, physics corrections, probes of results and live-battle QA in a greenfield project. Friction unrelated to context cost as much: the first era alone (January 7 to 11) had four usage-limit lockouts and a 19.4-hour permission block.

**Verdict: partly supported.** The 200k window shaped how Paul delegated (synchronous fan-out to save context, hourly compactions, plan-mode resets) and imposed a modest state-loss tax. Most of January's prompt volume had other causes. The later rise in autonomy is real, but models, harness features, project maturity and Paul's own experience changed at the same time, so the data cannot credit context size alone.

## Lessons

1. **Write the vision down before the first prompt.** A 13-line spec held for ten months.
2. **Run the thing before you accept it.** Green tests missed several of the defects above.
3. **Ask the agent to calculate or look it up,** which exposes invented constants.
4. **When you fix a bug, hunt for its siblings.**
5. **Make visual claims checkable,** by a person or by a snapshot harness plus a critic.
6. **Make every incident durable** as a memory note, a project rule or a test.
7. **Bound paid and long runs** with checkpoint caps, dollar caps and usage check-ins, and wait on completion notifications instead of polling.
8. **Design for quota interruptions,** so a killed workflow resumes its unfinished jobs and keeps the finished ones.
9. **Use independent reviewers, and verify the reviewers.**
10. **If you gate releases, also check that main is green.**

## Method

This history was reconstructed in October 2026 from the project's Claude Code session transcripts (including subagent and Workflow agent transcripts), the Claude Code prompt history and the git log, plus GitHub Actions runs for CI status. Several analysis passes covered the periods and the prompts, and a fact-check pass then matched every quote and date against those primary sources and recounted the core metrics with one rule. The numbers are estimates that move with the counting rule; the trends hold across rules. Blind spots: Claude Code on the web sessions (PRs #1 to #3) and two September 6 commits have no local transcript, so their steering can only be inferred from commits. Compaction replaces earlier context with a summary, so lost context is inferred from what Paul had to restate.
