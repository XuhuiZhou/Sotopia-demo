from typing import TypedDict

from sotopia.database import EpisodeLog


class messageForRendering(TypedDict):
    role: str
    type: str
    content: str


def parse_reasoning(reasoning: str, num_agents: int) -> tuple[list[str], str]:
    """Parse the reasoning string into a dictionary."""
    sep_token = "SEPSEP"
    for i in range(1, num_agents + 1):
        reasoning = (
            reasoning.replace(f"Agent {i} comments:\n", sep_token)
            .strip(" ")
            .strip("\n")
        )
    all_chunks = reasoning.split(sep_token)
    general_comment = all_chunks[0].strip(" ").strip("\n")
    comment_chunks = all_chunks[-num_agents:]

    return comment_chunks, general_comment


def render_for_humans(episode: EpisodeLog) -> list[messageForRendering]:
    """Generate a list of messages for human-readable version of the episode log."""

    messages_for_rendering: list[messageForRendering] = []

    for idx, turn in enumerate(episode.messages):
        is_observation_printed = False

        if idx == 0:
            assert (
                len(turn) >= 2
            ), "The first turn should have at least environment messages"

            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[0][2]}
            )
            messages_for_rendering.append(
                {"role": "Background Info", "type": "info", "content": turn[1][2]}
            )
            messages_for_rendering.append(
                {"role": "System", "type": "divider", "content": "Start Simulation"}
            )

        for sender, receiver, message in turn:
            if not is_observation_printed and "Observation:" in message and idx != 0:
                extract_observation = message.split("Observation:")[1].strip()
                if extract_observation:
                    messages_for_rendering.append(
                        {
                            "role": "Observation",
                            "type": "observation",
                            "content": extract_observation,
                        }
                    )
                is_observation_printed = True

            if receiver == "Environment":
                if sender != "Environment":
                    if "did nothing" in message:
                        continue
                    elif "left the conversation" in message:
                        messages_for_rendering.append(
                            {
                                "role": "Environment",
                                "type": "leave",
                                "content": f"{sender} left the conversation",
                            }
                        )
                    else:
                        if "said:" in message:
                            message = message.split("said:")[1].strip()
                            messages_for_rendering.append(
                                {"role": sender, "type": "said", "content": message}
                            )
                        else:
                            message = message.replace("[action]", "")
                            messages_for_rendering.append(
                                {"role": sender, "type": "action", "content": message}
                            )
                else:
                    messages_for_rendering.append(
                        {
                            "role": "Environment",
                            "type": "environment",
                            "content": message,
                        }
                    )

    messages_for_rendering.append(
        {"role": "System", "type": "divider", "content": "End Simulation"}
    )

    reasoning_per_agent, general_comment = parse_reasoning(
        episode.reasoning,
        len(
            set(
                msg["role"]
                for msg in messages_for_rendering
                if msg["type"] in {"said", "action"}
            )
        ),
    )

    if general_comment == "":
        return messages_for_rendering[:-1]

    messages_for_rendering.append(
        {"role": "General", "type": "comment", "content": general_comment}
    )

    for idx, reasoning in enumerate(reasoning_per_agent):
        messages_for_rendering.append(
            {
                "role": f"Agent {idx + 1}",
                "type": "comment",
                "content": f"{reasoning}\n\nRewards: {str(episode.rewards[idx])}",
            }
        )

    return messages_for_rendering
