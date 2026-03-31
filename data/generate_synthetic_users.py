# data/generate_synthetic_users.py
# Generates synthetic users with smoothly drifting interest timelines.
# Output: data/raw/synthetic_users.json

import json
import random
import numpy as np
from pathlib import Path

# ── output path ───────────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/raw/synthetic_users.json")

# ── topic graph ───────────────────────────────────────────────────────────────
# Each persona has one or more progression paths.
# Topics overlap between adjacent steps to create smooth drift.
TOPIC_GRAPH = {
    "ml": {
        "progressions": [
            ["python", "numpy", "pandas", "sklearn", "neural_nets",
             "keras", "pytorch", "transformers", "llms", "mlops"],
            ["python", "statistics", "pandas", "sklearn",
             "feature_engineering", "xgboost", "model_deployment", "mlops"],
            ["python", "numpy", "image_processing", "opencv",
             "cnns", "object_detection", "segmentation", "diffusion_models"],
        ]
    },
    "robotics": {
        "progressions": [
            ["python", "linear_algebra", "sensors", "control_systems",
             "ros", "kinematics", "path_planning", "slam", "reinforcement_learning"],
            ["python", "electronics", "arduino", "embedded_systems",
             "actuators", "pid_control", "ros", "robot_vision"],
        ]
    },
    "web": {
        "progressions": [
            ["html", "css", "javascript", "dom_manipulation",
             "react", "nodejs", "rest_apis", "databases", "devops", "cloud"],
            ["html", "css", "javascript", "typescript",
             "nextjs", "graphql", "prisma", "vercel", "web_performance"],
        ]
    },
    "data_science": {
        "progressions": [
            ["python", "pandas", "matplotlib", "seaborn",
             "sql", "tableau", "spark", "data_warehousing", "dbt"],
            ["python", "statistics", "hypothesis_testing",
             "r_language", "bayesian_inference", "a_b_testing", "causal_inference"],
        ]
    },
    "security": {
        "progressions": [
            ["networking", "linux", "bash", "python",
             "ethical_hacking", "ctf", "penetration_testing", "reverse_engineering"],
            ["networking", "cryptography", "python",
             "malware_analysis", "threat_intelligence", "incident_response"],
        ]
    }
}


def generate_user_timeline(
    user_id: str,
    persona: str,
    num_timesteps: int = 5,
    window_size: int = 3,      # topics visible at each step
    drift_speed: int = 2,      # steps forward in progression per timestep
    noise_prob: float = 0.10,  # chance of one off-topic noise topic
    seed_offset: int = 0,
) -> dict:
    """
    Generate one user with a smoothly drifting interest timeline.

    The window slides along the progression at drift_speed per timestep.
    Topics overlap between adjacent timesteps → smooth drift, not random jumps.
    Noise topics are added occasionally to simulate real-world curiosity.
    """
    progression = random.choice(TOPIC_GRAPH[persona]["progressions"])

    # Ensure the progression is long enough for the requested timesteps
    min_length = window_size + drift_speed * (num_timesteps - 1)
    while len(progression) < min_length:
        # Pad by repeating the last few topics (user has settled into an area)
        progression = progression + progression[-3:]

    timeline = []
    for t in range(num_timesteps):
        start = drift_speed * t
        end   = start + window_size
        current_topics = list(progression[start:end])

        # Occasionally add one noise topic from a different persona
        if random.random() < noise_prob:
            other_personas = [p for p in TOPIC_GRAPH if p != persona]
            noise_persona  = random.choice(other_personas)
            noise_prog     = random.choice(TOPIC_GRAPH[noise_persona]["progressions"])
            noise_topic    = random.choice(noise_prog[:3])  # only early/common topics
            current_topics.append(noise_topic)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for topic in current_topics:
            if topic not in seen:
                seen.add(topic)
                deduped.append(topic)

        timeline.append({
            "timestep": t,
            "topics":   deduped,
        })

    return {
        "user_id":  user_id,
        "persona":  persona,
        "timeline": timeline,
    }


def generate_dataset(
    num_users: int       = 50,
    num_timesteps: int   = 5,
    window_size: int     = 3,
    drift_speed: int     = 2,
    noise_prob: float    = 0.10,
    seed: int            = 42,
) -> list[dict]:

    random.seed(seed)
    np.random.seed(seed)

    personas = list(TOPIC_GRAPH.keys())
    users    = []

    for i in range(num_users):
        user_id = f"u{i+1:03d}"
        persona = personas[i % len(personas)]

        user = generate_user_timeline(
            user_id       = user_id,
            persona       = persona,
            num_timesteps = num_timesteps,
            window_size   = window_size,
            drift_speed   = drift_speed,
            noise_prob    = noise_prob,
            seed_offset   = i,
        )
        users.append(user)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(users, f, indent=2)

    print(f"Generated {len(users)} users → {OUTPUT_PATH}\n")
    _print_summary(users)
    return users


def _print_summary(users: list[dict]) -> None:
    from collections import Counter
    persona_counts = Counter(u["persona"] for u in users)

    print("Persona distribution:")
    for persona, count in persona_counts.items():
        print(f"  {persona:<20} {count} users")

    # Verify no empty topic lists
    empty = [(u["user_id"], s["timestep"])
             for u in users
             for s in u["timeline"]
             if len(s["topics"]) == 0]
    if empty:
        print(f"\nWARNING: {len(empty)} empty topic lists found: {empty}")
    else:
        print(f"\nNo empty topic lists — OK")

    # Show 2 example users
    for example in users[:2]:
        print(f"\nExample — {example['user_id']} ({example['persona']}):")
        for step in example["timeline"]:
            print(f"  t={step['timestep']}: {step['topics']}")


if __name__ == "__main__":
    generate_dataset(
        num_users     = 50,
        num_timesteps = 5,
        window_size   = 3,
        drift_speed   = 2,
        noise_prob    = 0.10,
        seed          = 42,
    )