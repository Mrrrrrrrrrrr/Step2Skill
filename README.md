# A Little Goes a Long Way: From Local Failures to Transferable GUI Skills

This repository is the anonymous code release for the CVPR submission **"A Little Goes a Long Way: From Local Failures to Transferable GUI Skills"**.

Step2Skill learns reusable multimodal GUI skills from localized failures. Given failed GUI interaction trajectories, it diagnoses the critical failure step, proposes and verifies local corrections without ground-truth corrective actions, consolidates accepted corrections into failure-conditioned multimodal skills, and recalls relevant skills during later task execution.

The paper evaluates Step2Skill on MobileGym multi-step task execution and on UI-Vision / MMBench-GUI single-step grounding. In the submitted version, Step2Skill improves MobileGym average success rate from 40.52% to 58.82% after three acquisition iterations, and transfers frozen skills to 80 unseen MobileGym tasks with a success-rate improvement from 45.83% to 54.58%.

## Repository Contents

```text
.
|-- sample.jsonl                        # Example task/category records
|-- run_mobilegym_flow.sh               # End-to-end MobileGym skill-learning workflow
|-- judge_trajs_parallel.sh             # Parallel trajectory diagnosis and judging
|-- classify_judge_results.sh           # Parallel failure-mode classification
|-- judge_repredict_steps_parallel.sh   # Parallel correction verification
|-- extract_skills_from_judgments.sh    # Skill extraction wrapper
|-- mobilegym_judge_run.py              # Converts MobileGym rollouts and diagnoses trajectories
|-- mobilegym_classify.py               # Maps failed trajectories to failure modes
|-- mobilegym_repredict.py              # Generates local correction candidates
|-- mobilegym_judge_step.py             # Verifies repredicted single-step corrections
|-- extract_skill.py                    # Builds/updates multimodal skill files
|-- count_*.py / count_*.sh             # Evaluation summaries
|-- api.py                              # OpenAI-compatible text/image model wrapper
`-- orm/
    |-- main.py                         # General ORM trajectory-judgment entry point
    |-- service.py                      # Optional HTTP service wrapper
    |-- config.py                       # Model and knowledge-base configuration
    |-- pipeline/                       # Pipeline presets used by ORM judging
    |-- prompts/                        # Prompt templates for diagnosis/judgment stages
    |-- knowledge_base/                 # Failure strategies and task knowledge
    `-- utils/                          # Evaluation, conversion, audit, and helper tools
```

## Method Overview

Step2Skill has two stages.

**Failure-to-Skill Acquisition** starts from raw GUI trajectories. It diagnoses whether a task failed, localizes the critical state-action error, assigns the failure to a failure mode, generates correction candidates through visual perturbation, cross-model, visual-feedback, or failure-conditioned proposals, verifies candidates with step-level multimodal judgments, and consolidates verified corrections into skills with applicability conditions, operational guidance, and visual examples.

**Failure-Aware Skill Reuse** runs during task execution. The base GUI agent proposes an action, Step2Skill recalls a relevant skill using the current GUI state and proposed action, and the agent re-decides when the recalled skill indicates a similar local failure condition.

## Environment

The code is Python-based with Bash workflow wrappers. A typical setup is:

```bash
conda create -n step2skill python=3.10 -y
conda activate step2skill
pip install requests urllib3 pillow flask openpyxl pandas opencv-python tqdm matplotlib
```

Additional dependencies depend on the components used:

- MobileGym and the base GUI agent are required for trajectory collection.
- A multimodal LLM endpoint compatible with the wrappers in `api.py` and `orm/utils/ask_qwen.py` is required for diagnosis, classification, correction generation, and verification.
- Bash is required for the `*.sh` orchestration scripts. On Windows, use WSL, Git Bash, or another Bash-compatible shell.

Before running experiments, configure model endpoints and credentials through environment variables or local configuration.

Common variables used by the scripts include:

```bash
export PYTHON_BIN=python
export MODEL_BASE_URL=http://YOUR_MODEL_SERVER/v1
export MOBILEGYM_MODEL_BASE_URL=http://YOUR_MOBILEGYM_AGENT_SERVER/v1
export API_KEY=YOUR_API_KEY
export WORKERS=8
```

## Data and Rollout Format

The MobileGym judging scripts accept either a MobileGym runs directory, one run directory, or one rollout directory. A rollout should contain files such as:

```text
rollout_dir/
|-- meta.json
|-- step_001_response.txt
|-- step_001.png / step_001.jpg
|-- step_001_annot.png / step_001_annot.jpg   # optional but useful for classification
|-- step_001_prompt.json                      # optional prompt messages
`-- ...
```

`sample.jsonl` gives a lightweight example of task records. Full MobileGym task generation and execution require the MobileGym environment and task set used by the evaluation protocol.

## Running the Full MobileGym Workflow

The top-level workflow is:

```bash
bash run_mobilegym_flow.sh
```

It performs the following steps:

1. Run MobileGym trajectories without skills.
2. Diagnose and judge trajectories.
3. Count baseline success rates.
4. Classify failed trajectories into failure modes.
5. Generate correction candidates for failed steps.
6. Verify repredicted step-level corrections.
7. Extract or update multimodal skills.
8. Run MobileGym trajectories with extracted skills.
9. Judge skill-augmented trajectories.
10. Count skill-augmented success rates.

The workflow uses these output directories by default:

```text
traj_0826/
judge_results_parallel_0826/
skills_0826/
traj_0826_skill/
judge_results_parallel_0826_skill/
```

You can override them with environment variables, for example:

```bash
TRAJ_DIR=runs/base \
JUDGE_DIR=outputs/base_judge \
SKILLS_DIR=outputs/skills \
SKILL_TRAJ_DIR=runs/skill \
SKILL_JUDGE_DIR=outputs/skill_judge \
WORKERS=8 \
bash run_mobilegym_flow.sh
```

## Running Individual Stages

Diagnose saved rollouts:

```bash
bash judge_trajs_parallel.sh path/to/trajs
```

This writes per-rollout results and a merged `mobilegym_skill_result.json` under `OUT_DIR`.

Classify failures:

```bash
RESULTS_DIR=outputs/base_judge \
OUTPUT_JSON=outputs/base_judge/mobilegym_error_classification.json \
bash classify_judge_results.sh outputs/base_judge/mobilegym_skill_result.json
```

Generate correction candidates:

```bash
python mobilegym_repredict.py \
  --input outputs/base_judge/mobilegym_error_classification.json \
  --output outputs/base_judge/mobilegym_edited_repredictions.json \
  --model-base-url "$MODEL_BASE_URL" \
  --api-key "$API_KEY"
```

Verify repredicted steps:

```bash
RESULTS_DIR=outputs/base_judge \
OUTPUT_JSON=outputs/base_judge/mobilegym_repredicted_step_judgments.json \
bash judge_repredict_steps_parallel.sh outputs/base_judge/mobilegym_edited_repredictions.json
```

Extract skills:

```bash
RESULTS_DIR=outputs/base_judge \
SKILLS_DIR=outputs/skills \
bash extract_skills_from_judgments.sh outputs/base_judge/mobilegym_repredicted_step_judgments.json
```

Summarize trajectory success:

```bash
bash count_mobilegym_orm_success.sh \
  --input outputs/skill_judge/mobilegym_skill_result.json \
  --manifest runs/skill/batch_runs.jsonl \
  --repeats 3 \
  --output-dir outputs/skill_judge
```

Summarize correction verification outputs:

```bash
python count_correct_corrections.py outputs/base_judge/judge_step_chunks/outputs
```

## ORM Utilities

The `orm/` directory contains the trajectory-judgment pipeline used by the MobileGym wrappers. It provides:

- pipeline presets under `orm/pipeline/`;
- prompt templates under `orm/prompts/`;
- task, slot, strategy, and failure knowledge under `orm/knowledge_base/`;
- conversion, evaluation, prompt-optimization, and audit utilities under `orm/utils/`.

An optional service wrapper can be started with:

```bash
cd orm
python service.py --host 0.0.0.0 --port 5000 --knowledge knowledge_base
```

For one-off rollout-folder prediction through the ORM utilities:

```bash
python orm/utils/predict_rollout_folder_orm.py path/to/rollout_folder \
  --pipeline-json orm/pipeline/pipeline_error_cause_analysis.json \
  --model seed_free \
  --knowledge knowledge_base \
  --output output.json
```

## Main Outputs

Typical intermediate and final outputs are:

- `mobilegym_skill_result.json`: trajectory-level task completion and localized failure diagnosis.
- `mobilegym_error_classification.json`: failure-mode classification for failed trajectories.
- `mobilegym_edited_repredictions.json`: local correction candidates and associated variants.
- `mobilegym_repredicted_step_judgments.json`: verifier judgments for correction candidates.
- `skills*/`: generated or updated multimodal skill files and visual examples.
- `success_summary.json` / CSV files from `count_*` scripts: task success and correction statistics.
