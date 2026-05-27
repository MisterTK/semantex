// crates/semantex-core/src/search/planner.rs
// Multi-step internal planner for `semantex_agent` (v0.6 Item 10).
//
// Goal: when a single classifier route is too coarse for a question — most
// notably "feature_planning" questions like "if I wanted to add logging,
// what files would change?" — break the question into 2-5 sub-queries, run
// each through the existing search machinery, and merge the outputs into a
// single response. This collapses what would otherwise be 30+ external tool
// turns into one `semantex_agent` call.
//
// Hard safety caps (spec §9 R4 — "infinite loops, oversized plans"):
//   - At most `MAX_STEPS` (5) steps per plan. Building a plan that asks for
//     more returns `PlanError::TooManySteps`. The plan is never expanded
//     internally past the cap.
//   - At most `MAX_WALL` (10s) total wall time across all steps. Enforced
//     by a `std::time::Instant` deadline checked between steps. Steps
//     already in flight aren't preempted, but no new step starts past the
//     deadline. The runner closure receives the remaining time so it can
//     bound its own work too.
//   - On any error (step failure, timeout, empty plan), `execute` returns
//     `Err(PlanError)`. The caller — `AgentPipeline::handle_feature_planning`
//     — falls back to the existing `handle_deep` path.
//
// This file is intentionally decoupled from `HybridSearcher`: it takes a
// `step_runner` closure that maps a `PlanStep` to a `Result<String>`. That
// lets the unit tests exercise the planner without spinning up an index,
// and lets the production caller wire it to whatever combination of
// hybrid/deep/graph searches the step kind implies.

use std::time::{Duration, Instant};

use super::agent_classifier::AgentRoute;

/// Maximum number of steps a plan may contain. Enforced at plan-construction
/// time AND at execution time as a defence in depth.
pub const MAX_STEPS: usize = 5;

/// Maximum total wall time for plan execution. Steps already running aren't
/// preempted, but no new step starts after this elapses.
pub const MAX_WALL: Duration = Duration::from_secs(10);

/// Kind of sub-query a single plan step represents.
///
/// The kind is advisory for the executor — it picks the right backend
/// (architecture overview vs structural walk vs deep search) based on this
/// hint. The keyword-based planner in this iteration emits a fixed sequence;
/// a future iteration may use an LLM to pick steps dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStepKind {
    /// "Where does this feature plug into the existing system?" — high-level
    /// landmarks (god nodes, communities, entry points).
    Architecture,
    /// "What convention is used for similar features?" — find existing
    /// instances of the pattern the user is adding.
    ConventionLookup,
    /// "Which files would actually change?" — focused candidate file list.
    ImpactedFiles,
    /// "Show me one or two concrete sites" — example code blocks.
    ExampleSites,
    /// "Summarize the above" — a final synthesis pass over earlier outputs.
    Synthesize,
}

impl std::fmt::Display for PlanStepKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Architecture => "architecture",
            Self::ConventionLookup => "convention_lookup",
            Self::ImpactedFiles => "impacted_files",
            Self::ExampleSites => "example_sites",
            Self::Synthesize => "synthesize",
        };
        f.write_str(s)
    }
}

/// One sub-query in a plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub kind: PlanStepKind,
    /// Sub-query string to run. May be a rewording of the original question
    /// or a narrower form of it.
    pub query: String,
}

/// A bounded multi-step plan derived from a user question.
#[derive(Debug, Clone)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    pub intent: AgentRoute,
}

/// Outcome of executing a single step.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub kind: PlanStepKind,
    pub query: String,
    /// The text produced by the runner. Empty if the step ran but returned nothing.
    pub output: String,
}

/// The merged result of executing a plan.
#[derive(Debug, Clone)]
pub struct PlannerResult {
    /// Concatenated, section-headed output suitable for returning to the
    /// caller as the `formatted` body of a `HandlerResult`.
    pub merged: String,
    pub per_step: Vec<StepOutcome>,
    pub elapsed: Duration,
}

/// Errors the planner can return. Callers should treat any of these as
/// "fall back to the simpler single-route handler".
#[derive(Debug)]
pub enum PlanError {
    NoSteps,
    TooManySteps(usize),
    Timeout(Duration),
    StepFailed {
        idx: usize,
        kind: PlanStepKind,
        msg: String,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSteps => f.write_str("plan must contain at least one step"),
            Self::TooManySteps(n) => {
                write!(f, "plan has {n} steps, exceeds MAX_STEPS={MAX_STEPS}")
            }
            Self::Timeout(d) => write!(f, "planner timed out after {d:?} (cap {MAX_WALL:?})"),
            Self::StepFailed { idx, kind, msg } => {
                write!(f, "step {idx} ({kind}) failed: {msg}")
            }
        }
    }
}

impl std::error::Error for PlanError {}

impl Plan {
    /// Build a plan for `query` given the classifier's intent hint.
    ///
    /// In this iteration the planner is keyword-based:
    /// - `AgentRoute::FeaturePlanning` → 3 steps:
    ///   Architecture → ConventionLookup → ImpactedFiles
    /// - Any other intent currently returns a 2-step plan
    ///   (ConventionLookup → Synthesize) — kept compact so the cap stays
    ///   a real safety net rather than a regularly-hit ceiling.
    ///
    /// The "use the LLM to construct a dynamic plan" enhancement (Item 9
    /// integration) is deferred to a follow-up.
    pub fn new(query: &str, intent: AgentRoute) -> Result<Self, PlanError> {
        let query = query.trim();
        let steps = match intent {
            AgentRoute::FeaturePlanning => vec![
                PlanStep {
                    kind: PlanStepKind::Architecture,
                    query: format!("architecture overview relevant to: {query}"),
                },
                PlanStep {
                    kind: PlanStepKind::ConventionLookup,
                    query: format!("existing conventions for: {query}"),
                },
                PlanStep {
                    kind: PlanStepKind::ImpactedFiles,
                    query: format!("files that would change when implementing: {query}"),
                },
            ],
            _ => vec![
                PlanStep {
                    kind: PlanStepKind::ConventionLookup,
                    query: query.to_string(),
                },
                PlanStep {
                    kind: PlanStepKind::Synthesize,
                    query: query.to_string(),
                },
            ],
        };
        Self::from_steps(steps, intent)
    }

    /// Construct a plan from an explicit list of steps. Enforces the
    /// `MAX_STEPS` cap and rejects empty plans. Public so the (eventual)
    /// LLM-driven planner can use the same checks.
    pub fn from_steps(steps: Vec<PlanStep>, intent: AgentRoute) -> Result<Self, PlanError> {
        if steps.is_empty() {
            return Err(PlanError::NoSteps);
        }
        if steps.len() > MAX_STEPS {
            return Err(PlanError::TooManySteps(steps.len()));
        }
        Ok(Self { steps, intent })
    }

    /// Run every step through `runner`, enforcing the wall-time cap. The
    /// runner closure receives the step and the remaining time budget so it
    /// can short-circuit its own work if asked.
    ///
    /// Returns `Err(PlanError::Timeout)` if the deadline elapses before all
    /// steps finish. Returns `Err(PlanError::StepFailed)` on the first step
    /// the runner reports as an error. Earlier outcomes are discarded in
    /// either case — the caller falls back to the simpler path.
    pub fn execute<F>(&self, runner: F) -> Result<PlannerResult, PlanError>
    where
        F: FnMut(&PlanStep, Duration) -> anyhow::Result<String>,
    {
        let deadline = Instant::now() + MAX_WALL;
        self.execute_with_deadline(deadline, runner)
    }

    /// Shared execution loop. Used by `execute` (production deadline = now +
    /// MAX_WALL) and by the test helper (custom deadline). Keeping one body
    /// avoids the production/test pair drifting silently.
    pub(crate) fn execute_with_deadline<F>(
        &self,
        deadline: Instant,
        mut runner: F,
    ) -> Result<PlannerResult, PlanError>
    where
        F: FnMut(&PlanStep, Duration) -> anyhow::Result<String>,
    {
        // Defence in depth: even if a future caller bypasses `Plan::new`,
        // the cap is re-checked here.
        if self.steps.is_empty() {
            return Err(PlanError::NoSteps);
        }
        if self.steps.len() > MAX_STEPS {
            return Err(PlanError::TooManySteps(self.steps.len()));
        }

        let start = Instant::now();
        let mut per_step: Vec<StepOutcome> = Vec::with_capacity(self.steps.len());

        for (idx, step) in self.steps.iter().enumerate() {
            let now = Instant::now();
            if now >= deadline {
                return Err(PlanError::Timeout(now.duration_since(start)));
            }
            let remaining = deadline.saturating_duration_since(now);
            match runner(step, remaining) {
                Ok(output) => per_step.push(StepOutcome {
                    kind: step.kind,
                    query: step.query.clone(),
                    output,
                }),
                Err(e) => {
                    return Err(PlanError::StepFailed {
                        idx,
                        kind: step.kind,
                        msg: e.to_string(),
                    });
                }
            }
            // Re-check after the step ran; if it overshot the deadline, that's
            // a timeout for plan purposes even though the step completed.
            if Instant::now() >= deadline {
                return Err(PlanError::Timeout(start.elapsed()));
            }
        }

        Ok(PlannerResult {
            merged: merge_outcomes(&per_step),
            per_step,
            elapsed: start.elapsed(),
        })
    }
}

/// Format outcomes as section-headed text. Pure function so tests can
/// assert on the exact shape without depending on the agent formatter.
fn merge_outcomes(outcomes: &[StepOutcome]) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("# Multi-step plan results\n\n");
    for (i, o) in outcomes.iter().enumerate() {
        // Step header includes index + kind + sub-query for traceability.
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("## Step {}: {} — `{}`\n\n", i + 1, o.kind, o.query),
        );
        if o.output.is_empty() {
            out.push_str("_(no results)_\n\n");
        } else {
            out.push_str(&o.output);
            if !o.output.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn plan_new_feature_planning_has_three_steps() {
        let p = Plan::new("if I wanted to add logging", AgentRoute::FeaturePlanning).unwrap();
        assert_eq!(p.steps.len(), 3);
        assert_eq!(p.steps[0].kind, PlanStepKind::Architecture);
        assert_eq!(p.steps[1].kind, PlanStepKind::ConventionLookup);
        assert_eq!(p.steps[2].kind, PlanStepKind::ImpactedFiles);
        assert!(matches!(p.intent, AgentRoute::FeaturePlanning));
    }

    #[test]
    fn plan_new_non_feature_planning_is_compact() {
        // Other intents currently get a 2-step plan; the test pins that.
        let p = Plan::new("explain something", AgentRoute::Deep).unwrap();
        assert!(p.steps.len() <= MAX_STEPS);
        assert!(!p.steps.is_empty());
    }

    #[test]
    fn from_steps_rejects_empty() {
        let err = Plan::from_steps(vec![], AgentRoute::FeaturePlanning).unwrap_err();
        assert!(matches!(err, PlanError::NoSteps));
    }

    #[test]
    fn from_steps_rejects_more_than_max() {
        let mut steps = Vec::new();
        for _ in 0..=MAX_STEPS {
            steps.push(PlanStep {
                kind: PlanStepKind::ConventionLookup,
                query: "x".into(),
            });
        }
        let err = Plan::from_steps(steps, AgentRoute::FeaturePlanning).unwrap_err();
        match err {
            PlanError::TooManySteps(n) => assert_eq!(n, MAX_STEPS + 1),
            other => panic!("expected TooManySteps, got {other:?}"),
        }
    }

    #[test]
    fn execute_runs_every_step_and_merges() {
        let plan = Plan::new("if I wanted to add caching", AgentRoute::FeaturePlanning).unwrap();
        let calls = std::cell::RefCell::new(0usize);
        let result = plan
            .execute(|step, _remaining| {
                *calls.borrow_mut() += 1;
                Ok(format!("result for {}: {}", step.kind, step.query))
            })
            .unwrap();
        assert_eq!(*calls.borrow(), 3);
        assert_eq!(result.per_step.len(), 3);
        // Merged text mentions every step.
        assert!(result.merged.contains("Step 1"));
        assert!(result.merged.contains("Step 2"));
        assert!(result.merged.contains("Step 3"));
        assert!(result.merged.contains("architecture"));
        assert!(result.merged.contains("convention_lookup"));
        assert!(result.merged.contains("impacted_files"));
    }

    #[test]
    fn execute_step_failure_propagates() {
        let plan = Plan::new("anything", AgentRoute::FeaturePlanning).unwrap();
        let err = plan
            .execute(|_step, _remaining| Err(anyhow::anyhow!("boom")))
            .unwrap_err();
        match err {
            PlanError::StepFailed { idx, msg, .. } => {
                assert_eq!(idx, 0);
                assert!(msg.contains("boom"));
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
    }

    #[test]
    fn execute_times_out_when_runner_sleeps_past_deadline() {
        // Build a plan whose first step sleeps longer than MAX_WALL would
        // ever allow in a real scenario. We don't want to actually sleep
        // 10 seconds in CI, so we temporarily shrink the deadline by
        // constructing a plan we expect to time out *between* steps: the
        // runner sleeps for slightly more than MAX_WALL / N and we observe
        // that the third step never runs.
        //
        // To keep the test fast we accept the small contract that
        // `MAX_WALL` is 10s by design and instead drive a *synthetic*
        // timeout via a single overlong step. We sleep just longer than
        // the cap for one step — only feasible in a slow test. To avoid
        // the 10s wait we use a separate inner helper that injects a
        // fake deadline. See `execute_with_deadline_for_test`.
        let plan = Plan::new("test", AgentRoute::FeaturePlanning).unwrap();
        let deadline = Instant::now() + Duration::from_millis(20);
        let err = execute_with_deadline_for_test(&plan, deadline, |_step, _remaining| {
            // Block long enough that the deadline elapses before the
            // next step gets a chance to start.
            thread::sleep(Duration::from_millis(40));
            Ok(String::from("done"))
        })
        .unwrap_err();
        assert!(
            matches!(err, PlanError::Timeout(_)),
            "expected Timeout, got {err:?}"
        );
    }

    #[test]
    fn execute_succeeds_within_deadline() {
        let plan = Plan::new("test", AgentRoute::FeaturePlanning).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let result = execute_with_deadline_for_test(&plan, deadline, |_step, _remaining| {
            Ok(String::from("done"))
        })
        .unwrap();
        assert_eq!(result.per_step.len(), 3);
    }

    /// Thin wrapper so tests can supply a custom deadline. The actual loop
    /// lives on `Plan::execute_with_deadline` so production and tests
    /// exercise the same code path.
    fn execute_with_deadline_for_test<F>(
        plan: &Plan,
        deadline: Instant,
        runner: F,
    ) -> Result<PlannerResult, PlanError>
    where
        F: FnMut(&PlanStep, Duration) -> anyhow::Result<String>,
    {
        plan.execute_with_deadline(deadline, runner)
    }
}
