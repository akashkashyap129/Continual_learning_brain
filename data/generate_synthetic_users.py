# data/generate_synthetic_users.py

import json
import random
import numpy as np
from pathlib import Path

TOPIC_GRAPH = {
    "ml": {
        "seed": ["python", "numpy", "math"],
        "progressions": [
            ["python", "numpy", "pandas", "sklearn", "neural_nets", "keras", "transformers", "llms"],
            ["python", "statistics", "sklearn", "feature_engineering", "xgboost", "mlops"],
            ["python", "numpy", "computer_vision", "opencv", "cnns", "object_detection"],
        ]
    },
    "robotics": {
        "seed": ["python", "math", "physics"],
        "progressions": [
            ["python", "sensors", "control_systems", "ros", "kinematics", "path_planning", "slam"],
            ["python", "math", "embedded_systems", "arduino", "actuators", "pid_control"],
        ]
    },
    "web": {
        "seed": ["html", "css", "javascript"],
        "progressions": [
            ["html", "css", "javascript", "react", "nodejs", "databases", "rest_apis", "devops"],
            ["html", "css", "javascript", "typescript", "nextjs", "graphql", "cloud"],
        ]
    },
    "data_science": {
        "seed": ["python", "excel", "statistics"],
        "progressions": [
            ["python", "pandas", "matplotlib", "sql", "tableau", "spark", "data_warehousing"],
            ["python", "statistics", "r_language", "bayesian_inference", "a_b_testing", "causal_inference"],
        ]
    },
    "security": {
        "seed": ["networking", "linux", "python"],
        "progressions": [
            ["networking", "linux", "bash", "ethical_hacking", "ctf", "penetration_testing", "reverse_engineering"],
            ["networking", "cryptography", "python", "malware_analysis", "threat_intelligence"],
        ]
    }
}

def generate_user_timeline(
    user_id: str,
    persona: str,
    num_timesteps: int = 5,
    topics_per_step: int = 3,
    noise_prob: float = 0.15,
    drift_speed: int = 2
) -> dict:
    """
    Generate one user with a realistic evolving interest timeline.

    drift_speed: how many new topics are introduced per time step
    noise_prob:  probability of adding a random off-topic interest
                 (simulates real-world noise — someone briefly curious about something)
    """
    graph = TOPIC_GRAPH[persona]
    progression = random.choice(graph["progressions"])

    timeline = []
    prog_index = 0  # current position in the progression

    for t in range(num_timesteps):
        # Window: take topics_per_step items from current position in progression
        start = max(0, prog_index)
        end = min(len(progression), start + topics_per_step)
        current_topics = list(progression[start:end])

        # Add noise: occasionally include a random topic from a different persona
        if random.random() < noise_prob:
            other_personas = [p for p in TOPIC_GRAPH if p != persona]
            noise_persona = random.choice(other_personas)
            noise_pool = TOPIC_GRAPH[noise_persona]["seed"]
            current_topics.append(random.choice(noise_pool))

        # Shuffle slightly so order within a step doesn't carry false meaning
        random.shuffle(current_topics)

        timeline.append({
            "timestep": t,
            "topics": current_topics
        })

        # Advance the window (drift)
        prog_index += drift_speed

    return {
        "user_id": user_id,
        "persona": persona,
        "timeline": timeline
    }


def generate_dataset(
    num_users: int = 50,
    num_timesteps: int = 5,
    topics_per_step: int = 3,
    noise_prob: float = 0.15,
    seed: int = 42,
    output_path: str = "data/synthetic_users.json"
) -> list[dict]:
    """
    Generate a full dataset of synthetic users across all personas.
    Users are distributed roughly evenly across personas.
    """
    random.seed(seed)
    np.random.seed(seed)

    personas = list(TOPIC_GRAPH.keys())
    users = []

    for i in range(num_users):
        user_id = f"u{i+1:03d}"
        # Round-robin persona assignment so each persona is well-represented
        persona = personas[i % len(personas)]

        user = generate_user_timeline(
            user_id=user_id,
            persona=persona,
            num_timesteps=num_timesteps,
            topics_per_step=topics_per_step,
            noise_prob=noise_prob
        )
        users.append(user)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(users, f, indent=2)

    print(f"Generated {len(users)} users → {output_path}")
    _print_summary(users)
    return users


def _print_summary(users: list[dict]) -> None:
    from collections import Counter
    persona_counts = Counter(u["persona"] for u in users)
    print("\nPersona distribution:")
    for persona, count in persona_counts.items():
        print(f"  {persona:<20} {count} users")

    # Show one example user
    example = users[0]
    print(f"\nExample — {example['user_id']} ({example['persona']}):")
    for step in example["timeline"]:
        print(f"  t={step['timestep']}: {step['topics']}")


if __name__ == "__main__":
    generate_dataset(
        num_users=50,
        num_timesteps=5,
        topics_per_step=3,
        noise_prob=0.15,
        seed=42
    )