import asyncio
from functools import wraps
from typing import cast

import streamlit as st
from sotopia.agents import Agents, LLMAgent
from sotopia.database import AgentProfile, EnvAgentComboStorage, EnvironmentProfile
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)


def async_to_sync(async_func):
    @wraps(async_func)
    def sync_func(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_func(*args, **kwargs))
        loop.close()
        return result

    return sync_func


MODEL = "gpt-4o-mini"
HUMAN_AGENT_IDX = 0


class ActionState:
    IDLE = 1
    HUMAN_WAITING = 2
    HUMAN_SPEAKING = 3
    MODEL_WAITING = 4
    MODEL_SPEAKING = 4
    EVALUATION_WAITING = 5


def print_current_speaker():
    if st.session_state.state == ActionState.HUMAN_SPEAKING:
        print("Human is speaking...")
    elif st.session_state.state == ActionState.MODEL_SPEAKING:
        print("Model is speaking...")
    elif st.session_state.state == ActionState.HUMAN_WAITING:
        print("Human is waiting...")
    elif st.session_state.state == ActionState.EVALUATION_WAITING:
        print("Evaluation is waiting...")
    else:
        print("Idle...")


conversation_container = st.empty()
input_container = st.empty()


def pretty_print_messages(messages):
    messages = [
        [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
        for messages_in_turn in messages
    ]
    messages_and_rewards = []
    for idx, turn in enumerate(messages):
        messages_in_this_turn = []
        if idx == 0:
            assert (
                len(turn) >= 2
            ), "The first turn should have at least environment messages"
            messages_in_this_turn.append(turn[0][2])
            messages_in_this_turn.append(turn[1][2])
        for sender, receiver, message in turn:
            if receiver == "Environment":
                if sender != "Environment":
                    if "did nothing" in message:
                        continue
                    else:
                        if "said:" in message:
                            messages_in_this_turn.append(f"{sender} {message}")
                        else:
                            messages_in_this_turn.append(f"{sender}: {message}")
                else:
                    messages_in_this_turn.append(message)
        messages_and_rewards.append("\n".join(messages_in_this_turn))

    for idx, message in enumerate(messages_and_rewards):
        st.write(message)
        st.write("")


def get_full_name(agent_profile: AgentProfile):
    return f"{agent_profile.first_name} {agent_profile.last_name}"


def initialize_session_state() -> None:
    if "active" not in st.session_state:
        st.session_state.active = False
        # if not st.session_state.active:
        st.session_state.conversation = []
        st.session_state.background = "Default Background"
        st.session_state.env_agent_combo = EnvAgentComboStorage.get(
            list(EnvAgentComboStorage.all_pks())[0]
        )
        st.session_state.state = ActionState.IDLE
        st.session_state.env = None
        st.session_state.agents = None
        st.session_state.environment_messages = None
        st.session_state.messages = []
        st.session_state.rewards = []
        st.session_state.reasons = []

    all_agents = [AgentProfile.get(id) for id in list(AgentProfile.all_pks())[:10]]
    all_envs = [
        EnvironmentProfile.get(id) for id in list(EnvironmentProfile.all_pks())[:10]
    ]

    st.session_state.agent_mapping = [
        {get_full_name(agent_profile): agent_profile for agent_profile in all_agents}
    ] * 2
    st.session_state.env_mapping = {
        env_profile.codename: env_profile for env_profile in all_envs
    }


def reset_session_state():
    pass


def get_env_agents(env_agent_combo) -> None:
    environment_profile = EnvironmentProfile.get(pk=env_agent_combo.env_id)
    agent_profiles = [
        AgentProfile.get(pk=agent_id) for agent_id in env_agent_combo.agent_ids
    ]
    agent_list = [
        LLMAgent(agent_profile=agent_profile, model_name=MODEL)
        for agent_profile in agent_profiles
    ]

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env = ParallelSotopiaEnv(
        action_order="round-robin",
        model_name=MODEL,
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(
                "gpt-4o",
                EvaluationForTwoAgents[SotopiaDimensions],
            ),
        ],
        env_profile=environment_profile,
    )

    environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()

    return env, agents, environment_messages


POSITION_CHOICES = ["First Agent", "Second Agent"]


def chat_demo() -> None:
    """
    State machine: HUMAN_WAITING (waiting for human input) -> HUMAN_SPEAKING -> MODEL_WAITING == MODEL_SPEAKING (model generate) -> HUMAN_WAITING
    Use "active" to indicate whether the conversation is still ongoing
    Adapted from sotopia.server.arun_one_episode

    # TODO manually specify the agent goals
    """

    initialize_session_state()

    scenarios = st.session_state.env_mapping
    agent_list_1, agent_list_2 = st.session_state.agent_mapping

    scenario_choice = st.selectbox(
        "Choose a scenario:", scenarios.keys(), disabled=st.session_state.active
    )

    agent_col1, agent_col2 = st.columns(2)
    with agent_col1:
        agent_choice_1 = st.selectbox(
            "Choose the first agent:",
            agent_list_1.keys(),
            disabled=st.session_state.active,
        )
    with agent_col2:
        agent_choice_2 = st.selectbox(
            "Choose the second agent:",
            agent_list_2.keys(),
            disabled=st.session_state.active,
        )
    user_position = st.selectbox(
        "Do you want to be the first or second agent?",
        POSITION_CHOICES,
        disabled=st.session_state.active,
    )
    start_button = st.button("Start", disabled=st.session_state.active)
    HUMAN_AGENT_IDX = 0 if user_position == "First Agent" else 1

    env_agent_combo = EnvAgentComboStorage(
        env_id=scenarios[scenario_choice].pk,
        agent_ids=[agent_list_1[agent_choice_1].pk, agent_list_2[agent_choice_2].pk],
    )

    user_input = st.text_input(
        "Your message:",
        key="user_input",
        disabled=st.session_state.state != ActionState.HUMAN_WAITING,
    )
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        send_button = st.button(
            "Send", disabled=st.session_state.state != ActionState.HUMAN_WAITING
        )
    with button_col2:
        stop_button = st.button("Stop", disabled=not st.session_state.active)

    env, agents, environment_messages = get_env_agents(env_agent_combo)

    if start_button:
        print("Starting conversation...")
        st.session_state.state = (
            ActionState.MODEL_SPEAKING
            if HUMAN_AGENT_IDX == 1
            else ActionState.HUMAN_WAITING
        )
        st.session_state.env = env
        st.session_state.agents = agents
        st.session_state.environment_messages = environment_messages
        st.session_state.active = True

    if send_button and st.session_state.state == ActionState.HUMAN_WAITING:
        st.session_state.human_input = user_input
        st.session_state.state = ActionState.HUMAN_SPEAKING

    if stop_button:
        print("Stopping conversation...")
        st.session_state.state = ActionState.EVALUATION_WAITING

    messages = [
        ("Environment", agent_name, environment_messages[agent_name])
        for agent_name in env.agents
    ]
    background = "\n".join(
        [f"{agent_name}: {message}" for _, agent_name, message in messages]
    ).replace("\n", "<br />")

    st.title("Chat with a model")
    st.markdown(
        f"""
        <details>
            <summary>Background</summary>
            {background}
        </details>
        """,
        unsafe_allow_html=True,
    )

    for index, agent_name in enumerate(env.agents):
        agents[agent_name].goal = env.profile.agent_goals[index]

    done = False

    if st.session_state.messages == []:
        st.session_state.messages.append(
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )

    print_current_speaker()
    print("State: ", st.session_state.state)
    print("Active: ", st.session_state.active)

    if st.session_state.active and st.session_state.state in [
        ActionState.HUMAN_SPEAKING,
        ActionState.MODEL_SPEAKING,
        ActionState.EVALUATION_WAITING,
    ]:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()

        actions = []
        for agent_idx, agent_name in enumerate(env.agents):
            if agent_idx == HUMAN_AGENT_IDX:
                # if this is the human's turn (actually this is determined by the action_mask)
                match st.session_state.state:
                    case ActionState.HUMAN_SPEAKING:
                        action = AgentAction(action_type="speak", argument=user_input)
                    case ActionState.EVALUATION_WAITING:
                        action = AgentAction(action_type="leave", argument="")
                    case _:
                        action = AgentAction(action_type="none", argument="")

                print("Human output action: ", action)
            else:
                match st.session_state.state:
                    case ActionState.HUMAN_SPEAKING:
                        action = AgentAction(action_type="none", argument="")
                    case ActionState.MODEL_SPEAKING:
                        action = async_to_sync(
                            st.session_state.agents[agent_name].aact
                        )(st.session_state.environment_messages[agent_name])
                    case ActionState.EVALUATION_WAITING:
                        action = AgentAction(action_type="leave", argument="")
                print("Model output action: ", action)

            actions.append(action)
        actions = cast(list[AgentAction], actions)

        for idx, agent_name in enumerate(st.session_state.env.agents):
            agent_messages[agent_name] = actions[idx]
            st.session_state.messages[-1].append(
                (agent_name, "Environment", agent_messages[agent_name])
            )

        # send agent messages to environment
        (
            st.session_state.environment_messages,
            rewards_in_turn,
            terminated,
            ___,
            info,
        ) = async_to_sync(st.session_state.env.astep)(agent_messages)
        st.session_state.messages.append(
            [
                (
                    "Environment",
                    agent_name,
                    st.session_state.environment_messages[agent_name],
                )
                for agent_name in st.session_state.env.agents
            ]
        )

        done = all(terminated.values())

        if done:
            print("Conversation ends...")
            st.session_state.state = ActionState.IDLE
            st.session_state.active = False
            st.session_state.done = False

            # from sotopia.database import EpisodeLog
            # agent_list = list(st.session_state.agents.values())
            # epilog = EpisodeLog(
            #     environment=env.profile.pk,
            #     agents=[agent.profile.pk for agent in agent_list],
            #     tag="tmp",
            #     models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
            #     messages=[
            #         [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            #         for messages_in_turn in messages
            #     ],
            #     reasoning=info[env.agents[0]]["comments"],
            #     rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
            #     rewards_prompt=info["rewards_prompt"]["overall_prompt"],
            # )
            st.session_state.rewards = [
                info[agent_name]["complete_rating"]
                for agent_name in st.session_state.env.agents
            ]
            st.session_state.reasoning = info[st.session_state.env.agents[0]][
                "comments"
            ]
            st.session_state.rewards_prompt = info["rewards_prompt"]["overall_prompt"]

        match st.session_state.state:
            case ActionState.HUMAN_SPEAKING:
                st.session_state.state = ActionState.MODEL_WAITING
            case ActionState.MODEL_SPEAKING:
                st.session_state.state = ActionState.HUMAN_WAITING
            case ActionState.EVALUATION_WAITING:
                st.session_state.state = ActionState.IDLE
                st.session_state.active = False
            case ActionState.IDLE:
                st.session_state.state = ActionState.IDLE
            case _:
                raise ValueError("Invalid state", st.session_state.state)

        st.rerun()

    if st.session_state.messages:
        st.write("Conversation History:")
        pretty_print_messages(st.session_state.messages)

        if (
            st.session_state.rewards != []
            and st.session_state.state == ActionState.IDLE
        ):
            st.write("Rewards:")
            for idx, reward in enumerate(st.session_state.rewards):
                st.write(f"Agent {idx}: {reward}")
            st.write("Reasons:")
            st.write(st.session_state.reasoning)
