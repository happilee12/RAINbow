import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from evaluation import evaluate_submission


HF_TOKEN = os.environ.get("HF_TOKEN")
SUBMISSIONS_REPO = os.environ.get("SUBMISSIONS_REPO")
RESULTS_REPO = os.environ.get("RESULTS_REPO")
REFERENCE_REPO = os.environ.get("REFERENCE_REPO")

ENV_INFO_SUBDIR = ""
RESULTS_FILE = "leaderboard.csv"

LEADERBOARD_COLUMNS = [
    "team_name",
    "method_name",
    
    "sr_test",
    "sr_unseen",
    "sr_seen",

    "dtc_test",
    "dtc_unseen",
    "dtc_seen",

    "spl_test",
    "spl_unseen",
    "spl_seen",

    "oracle_sr_test",
    "oracle_sr_unseen",
    "oracle_sr_seen",

    "steps_test",
    "steps_unseen",
    "steps_seen",

    # "count_test",
    # "count_unseen",
    # "count_seen",

    "submitted_at",
]

NUMERIC_COLUMNS = [
    "sr_test",
    "sr_unseen",
    "sr_seen",

    "dtc_test",
    "dtc_unseen",
    "dtc_seen",

    "spl_test",
    "spl_unseen",
    "spl_seen",

    "oracle_sr_test",
    "oracle_sr_unseen",
    "oracle_sr_seen",

    "steps_test",
    "steps_unseen",
    "steps_seen",

    # "count_test",
    # "count_unseen",
    # "count_seen",
]

api = HfApi(token=HF_TOKEN)


def get_uploaded_file_path(submission_file) -> str:
    """
    Gradio File output can be either a path-like string or an object with `.name`,
    depending on the Gradio version/config.
    """
    if submission_file is None:
        raise ValueError("No submission file was uploaded.")

    if isinstance(submission_file, str):
        return submission_file

    if hasattr(submission_file, "name"):
        return submission_file.name

    raise ValueError("Could not read the uploaded file path.")


@lru_cache(maxsize=1)
def get_env_info_dir() -> str:
    """
    Download env_info files from the reference dataset repo and return local env_info path.

    Expected reference repo structure:
    REFERENCE_REPO/
        shortest_distances_by_split.json
        shortest_paths_by_split.json
        episode_metadata_by_split.json
    """
    if not REFERENCE_REPO:
        raise ValueError("REFERENCE_REPO is not configured.")

    repo_dir = snapshot_download(
        repo_id=REFERENCE_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns=[
            "shortest_distances_by_split.json",
            "shortest_paths_by_split.json",
            "episode_metadata_by_split.json",
            "scans_by_split.json",
        ],
    )

    env_info_dir = Path(repo_dir)

    required_files = [
        "shortest_distances_by_split.json",
        "shortest_paths_by_split.json",
        "episode_metadata_by_split.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (env_info_dir / filename).exists()
    ]

    if missing_files:
        available_files = [p.name for p in env_info_dir.glob("*")] if env_info_dir.exists() else []

        raise FileNotFoundError(
            "Missing required env_info file(s) in REFERENCE_REPO root.\n"
            f"Missing: {missing_files}\n"
            f"Downloaded repo path: {env_info_dir}\n"
            f"Available files: {available_files}\n"
            "Expected files at repo root: "
            "shortest_distances_by_split.json, "
            "shortest_paths_by_split.json, "
            "episode_metadata_by_split.json."
        )

    return str(env_info_dir)

def load_leaderboard() -> pd.DataFrame:
    """Load leaderboard.csv from the results dataset repo."""
    if not RESULTS_REPO:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    try:
        path = hf_hub_download(
            repo_id=RESULTS_REPO,
            filename=RESULTS_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        df = pd.read_csv(path)

        for col in LEADERBOARD_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[LEADERBOARD_COLUMNS]
        return df

    except Exception:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)


def save_leaderboard(df: pd.DataFrame):
    """Upload leaderboard.csv to the results dataset repo."""
    if not RESULTS_REPO:
        raise ValueError("RESULTS_REPO is not configured.")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, RESULTS_FILE)
        df.to_csv(local_path, index=False)

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=RESULTS_FILE,
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message="Update DialNav leaderboard",
        )


def save_submission_file(team_name: str, submission_file):
    """Save raw uploaded submission file to the submissions dataset repo."""
    if not SUBMISSIONS_REPO:
        raise ValueError("SUBMISSIONS_REPO is not configured.")

    file_path = get_uploaded_file_path(submission_file)

    safe_team = (
        team_name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    original_name = Path(file_path).name
    path_in_repo = f"{safe_team}/{timestamp}_{original_name}"

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=SUBMISSIONS_REPO,
        repo_type="dataset",
        commit_message=f"Add submission from {team_name}",
    )

    return path_in_repo


def run_dialnav_evaluation(submission_file):
    """
    Run official DialNav evaluator on one split-keyed submission JSON.

    Expected submission.json:
    {
      "val_seen": [...],
      "val_unseen": [...],
      "test": [...]
    }
    """
    env_info_dir = get_env_info_dir()
    submission_path = get_uploaded_file_path(submission_file)

    scores = evaluate_submission(
        submission_file=submission_path,
        env_info_dir=env_info_dir,
        success_margin=0,
        error_margin=3.0,
    )

    return scores



def sort_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(
        by=["sr_test", "sr_unseen", "sr_seen"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def build_result_row(team_name, method_name, scores, submitted_at):
    """
    Build one leaderboard row from evaluator output.
    Final score and split-level score columns are intentionally not displayed
    on the public leaderboard.
    """


    row = {
        "sr_test": scores.get("sr_test", 0.0),
        "sr_unseen": scores.get("sr_unseen", 0.0),
        "sr_seen": scores.get("sr_seen", 0.0),

        "dtc_test": scores.get("dtc_test", 0.0),
        "dtc_unseen": scores.get("dtc_unseen", 0.0),
        "dtc_seen": scores.get("dtc_seen", 0.0),

        "spl_test": scores.get("spl_test", 0.0),
        "spl_unseen": scores.get("spl_unseen", 0.0),
        "spl_seen": scores.get("spl_seen", 0.0),

        "oracle_sr_test": scores.get("oracle_sr_test", 0.0),
        "oracle_sr_unseen": scores.get("oracle_sr_unseen", 0.0),
        "oracle_sr_seen": scores.get("oracle_sr_seen", 0.0),

        "steps_test": scores.get("steps_test", 0.0),
        "steps_unseen": scores.get("steps_unseen", 0.0),
        "steps_seen": scores.get("steps_seen", 0.0),

        "count_test": scores.get("count_test", 0),
        "count_unseen": scores.get("count_unseen", 0),
        "count_seen": scores.get("count_seen", 0),

        "team_name": team_name.strip(),
        "method_name": method_name.strip() if method_name else "",
        "submitted_at": submitted_at,
    }

    return row
    

def submit(
    team_name,
    contact_email,
    method_name,
    submission_file,
    agreement,
):
    if not agreement:
        return "Please confirm the agreement before submitting.", load_leaderboard()

    if not team_name or not team_name.strip():
        return "Please enter your team name.", load_leaderboard()

    if not contact_email or not contact_email.strip():
        return "Please enter your contact email.", load_leaderboard()

    if submission_file is None:
        return "Please upload submission.json.", load_leaderboard()

    try:
        saved_path = save_submission_file(team_name, submission_file)
        scores = run_dialnav_evaluation(submission_file)

        submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        new_row = build_result_row(
            team_name=team_name,
            method_name=method_name,
            scores=scores,
            submitted_at=submitted_at,
        )

        df = load_leaderboard()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Keep the best score per team.
        df = sort_leaderboard(df)
        # df = df.drop_duplicates(subset=["team_name"], keep="first")
        # df = sort_leaderboard(df)

        save_leaderboard(df)

        message = (
            "Submission received successfully.\n\n"
            f"Saved submission path: `{saved_path}`\n\n"
            "Scores:\n\n"
            f"- SR Test: **{float(scores.get('sr_test', 0.0)):.2f}**\n"
            f"- SR Unseen: {float(scores.get('sr_unseen', 0.0)):.2f}\n"
            f"- SR Seen: {float(scores.get('sr_seen', 0.0)):.2f}\n"
        )

        return message, df

    except Exception as e:
        return f"Submission failed: {str(e)}", load_leaderboard()


def refresh_leaderboard():
    df = load_leaderboard()
    return sort_leaderboard(df)



ABOUT_TEXT = """
# DialNav Challenge

Welcome to the DialNav Challenge evaluation server.

The DialNav Challenge benchmarks dialog-enabled embodied navigation agents. In each episode, an agent starts from a given location and must navigate to a target location by interacting with a guide through natural language dialog.

Participants may modify the navigator, the guide, or both components. The goal is to improve navigation success by using dialog effectively to resolve ambiguity, recover from uncertainty, and reach the target location more reliably.

This challenge is organized as an open benchmark rather than a hidden-test competition. Since the guide component may require access to target information in order to provide meaningful assistance, all input splits, including the test split, may be released to participants.

## Stay Updated

Please fill out the following form to stay updated on DialNav Challenge announcements and important deadlines: https://tally.so/r/81rQbz


## Evaluation Protocol

Detailed evaluation protocol and scoring rules will be announced later.
"""

SUBMIT_TEXT = """
## Submit Your DialNav Results

Please upload a single JSON submission file.

Before submitting, please make sure that:

- your contact email is valid: the challenge organizers will reach out for prizes and request for technical report with this email
- your submission file follows the official format: https://drive.google.com/file/d/1ZJAql237FmqNbwX-4Mt9ymTOVbdd8njn/view?usp=sharing
- your submission contains predictions/logs for all required episodes
- communication between the navigator and the guide is carried out only through natural language
- you do not tune model parameters, prompts, thresholds, or hyperparameters on the test dataset.

Stay updated for news here: https://tally.so/r/81rQbz

Submissions are unlimited. However, participants are expected to use the evaluation server responsibly.
"""
FORMAT_TEXT = """

Submission Format

Please upload a single submission.json file.

You may refer to the following sample submission file when preparing your submission:

Sample Submission File: https://drive.google.com/file/d/1ZJAql237FmqNbwX-4Mt9ymTOVbdd8njn/view?usp=sharing

The top-level JSON object must contain three keys, one list per split:

~~~json
{
  "val_seen": [],
  "val_unseen": [],
  "test": []
}
~~~

Each split key must map to a list of episode-level navigation outputs.

Example:

~~~json
{
    "val_seen": [
        {
            "instr_id": 190,
            "scan": "5q7pvUzZiYa",
            "path": [
                "7dc12a67ddfc4a4a849ce620db5b777b",
                "0e84cf4dec784bc28b78a80bee35c550",
                "a77784b955454209857d745976a1676d",
                "67971a17c26f4e2ca117b4fca73507fe",
                "8db06d3a0dd44508b3c078d60126ce19",
                "43ac37dfa1db4a13a8a9df4e454eb016",
                "4bd82c990a6548a994daa97c8f52db06",
                "6d11ca4d41e04bb1a725c2223c36b2aa",
                "29fb3c58b29348558d36a9f9440a1379",
                "c23f26401359426982d11ca494ee739b",
                "397403366d784caf804d741f32fd68b9",
                "3c6a35e15ada4b649990d6568cce8bd9",
                "55e4436f528c4bf09e4550079c572f7b",
                "c629c7f1cf6f47a78c45a8ae9ff82247",
                "a7f591c0bf5443a8989e6adf9968cf90",
                "a0a40af004954a90b2b24a0cc6655881",
                "24e8ebf554d742069e81afd7eb3369c7",
                "4214a9b1607e4cbea8dcdb42a4952796",
                "ed79b51efdac434cb12bcd3d5e7fd112",
                "82e8acfab99d4441ac4da5bf22c2642d",
                "8836420cacbe4c37a57a8f8645da320b",
                "041c71a6b7364371892021424389107a",
                "19e0d074a48044949855917a5bff42ad",
                "6be22e287d2f4bb4a5fd1fc1172d438b",
                "7939ba3afc934e268aec59027819ae4b",
                "ffbed768a549496c9752a736bff9efd5"
            ],
            "dialog": [
                {
                    "nav_idx": 12,
                    "question": "i 'm in a modern space filled with art , and there 's a glass railing staircase along with a painting of marilyn monroe on the wall . ",
                    "answer": "perfect ! now , go down the stairs and take a left . walk straight ahead , then turn left again to enter the living room . after that , turn slightly left and stand next to the sofa . you 've made it to the living room ! ",
                    "localized_viewpoint": "55e4436f528c4bf09e4550079c572f7b",
                    "viewpoint": "55e4436f528c4bf09e4550079c572f7b"
                }
            ]
        },
        ...
  ],
  "val_unseen": [
    ...
  ],
  "test": [
    ...
  ]
}
~~~

Required top-level keys:

- `val_seen`
- `val_unseen`
- `test`

Required item-level fields:

- `instr_id`
- `path`
- `dialog`

Each dialog item should include:

- `question`
- `viewpoint`
- `localized_viewpoint`
- `answer`
- `nav_idx`

## Important

All required fields must be included, even if some fields are not directly used for the final score computation. These fields are necessary for format validation, episode matching, reproducibility, and consistency across submissions.

Submissions that do not follow the official format may be rejected or considered invalid. This includes missing required split keys, missing required item-level fields, incorrectly formatted fields, or outputs that cannot be matched to the official episode metadata.

"""



FAQ_TEXT = """
# Rules and FAQ

## How can we stay updated?

Please fill out the stay-updated form below to receive challenge-related announcements and important deadline updates: https://tally.so/r/81rQbz

## Can we modify the navigator?

Yes. Participants may modify the navigator.

## Can we modify the guide?

Yes. Participants may modify the guide.

## Can we modify both the navigator and the guide?

Yes. Participants may modify the navigator, the guide, or both components.

## Is this a hidden-test competition?

No. The DialNav Challenge is organized as an open benchmark for reproducible comparison rather than a hidden-test competition.

## Can the guide access target information?

Yes. The guide may use target information when generating assistance for the navigator. However, communication between the navigator and the guide must be carried out only through natural language.

## Can the navigator and guide communicate using images or other non-language signals?

No. Communication between the navigator and the guide must be natural-language only. Images or any other non-language communication channels between the navigator and the guide are not allowed.

## Can we tune on the test dataset?

No. The test dataset should be used for final evaluation only. Participants must not tune model parameters, prompts, thresholds, hyperparameters, or method choices based on test-set performance.

## Are external datasets, pretrained models, LLMs, VLMs, or commercial APIs allowed?

Yes. They are allowed, but participants may be asked to report them for transparency.

## Is there a submission limit?

No. There is no submission limit. However, participants are expected to use the evaluation server responsibly.

## How many members can a team have?

Each team may include up to 5 members. Each participant may belong to only one team.

## Who should we contact if we have questions?

For questions about the DialNav Challenge, please contact the organizers at: happilee12@korea.ac.kr
"""


with gr.Blocks(title="DialNav Challenge") as demo:
    gr.Markdown("# 🧭 DialNav Challenge Evaluation Server")

    with gr.Tab("About"):
        gr.Markdown(ABOUT_TEXT)

    with gr.Tab("Leaderboard"):
        gr.Markdown("## Leaderboard")
        refresh_button = gr.Button("Refresh Leaderboard")
        leaderboard_table = gr.Dataframe(
            value=refresh_leaderboard(),
            interactive=False,
            wrap=True,
        )
        refresh_button.click(
            fn=refresh_leaderboard,
            inputs=[],
            outputs=[leaderboard_table],
        )

    with gr.Tab("Submit"):
        gr.Markdown(SUBMIT_TEXT)

        team_name = gr.Textbox(
            label="Team name",
            placeholder="Use the exact team name from the registration form.",
        )
        contact_email = gr.Textbox(
            label="Contact email",
            placeholder="This will not be displayed publicly.",
        )
        method_name = gr.Textbox(
            label="Method name",
            placeholder="Optional",
        )
        submission_file = gr.File(
            label="Upload submission.json",
            file_types=[".json"],
            type="filepath",
        )
        agreement = gr.Checkbox(
            label="I confirm that this submission follows the DialNav Challenge rules.",
            value=False,
        )

        submit_button = gr.Button("Submit")
        output_message = gr.Markdown()
        updated_leaderboard = gr.Dataframe(interactive=False, wrap=True)

        submit_button.click(
            fn=submit,
            inputs=[
                team_name,
                contact_email,
                method_name,
                submission_file,
                agreement,
            ],
            outputs=[output_message, updated_leaderboard],
        )

    with gr.Tab("Submission Format"):
        gr.Markdown(FORMAT_TEXT)

    with gr.Tab("Rules & FAQ"):
        gr.Markdown(FAQ_TEXT)


if __name__ == "__main__":
    demo.launch()