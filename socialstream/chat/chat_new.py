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
from sotopia.messages import AgentAction


def async_to_sync(async_func) -> callable:
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
POSITION_CHOICES = ["First Agent", "Second Agent"]


class ActionState:
    IDLE = 1
    HUMAN_WAITING = 2
    HUMAN_SPEAKING = 3
    MODEL_WAITING = 4
    MODEL_SPEAKING = 4
    EVALUATION_WAITING = 5


def print_current_speaker() -> None:
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


def get_full_name(agent_profile: AgentProfile) -> str:
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
        st.session_state.rewards = [0.0, 0.0]
        st.session_state.reasoning = ""

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


def step(user_input: str | None = None) -> None:
    # import time
    # time.sleep(5)
    # return
    env = st.session_state.env

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
                    action = async_to_sync(st.session_state.agents[agent_name].aact)(
                        st.session_state.environment_messages[agent_name]
                    )
                case ActionState.EVALUATION_WAITING:
                    action = AgentAction(action_type="leave", argument="")
                case _:
                    action = AgentAction(action_type="none", argument="")
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

        from sotopia.database import EpisodeLog

        agent_list = list(st.session_state.agents.values())

        st.session_state.rewards = [
            info[agent_name]["complete_rating"]
            for agent_name in st.session_state.env.agents
        ]
        st.session_state.reasoning = info[st.session_state.env.agents[0]]["comments"]
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

    print("State after step: ", st.session_state.state)
    done = all(terminated.values())


def get_env_agents(env_agent_combo):  # type: ignore
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


def chat_demo() -> None:
    initialize_session_state()
    print("Current state:", st.session_state.state)

    with st.expander("Scenario Settings"):
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

        if (
            agent_choice_1 or agent_choice_2 or user_position or scenario_choice
        ) and not st.session_state.active:
            print("Setting settings...")
            set_settings(agent_choice_1, agent_choice_2, scenario_choice, user_position)

    start_col, stop_col = st.columns(2)
    with start_col:
        start_button = st.button("Start", disabled=st.session_state.active)
        if start_button:
            st.session_state.active = True
            st.session_state.state = (
                ActionState.HUMAN_WAITING
                if user_position == "First Agent"
                else ActionState.MODEL_WAITING
            )

            if st.session_state.state == ActionState.MODEL_WAITING:
                with st.spinner("Model acting..."):
                    step()  # model's turn
            # st.rerun()

    with stop_col:
        stop_button = st.button("Stop", disabled=not st.session_state.active)
        if stop_button:
            st.session_state.active = False
            st.session_state.state = ActionState.EVALUATION_WAITING

    with st.form("user_input", clear_on_submit=True):
        user_input = st.text_input("Enter your message here:", key="user_input")
        if st.form_submit_button(
            "Submit",
            use_container_width=True,
            disabled=st.session_state.state != ActionState.HUMAN_WAITING,
        ):
            with st.spinner("Human Acting..."):
                st.session_state.state = ActionState.HUMAN_SPEAKING
                print_current_speaker()
                step(user_input=user_input)  # human's turn
            with st.spinner("Model Acting..."):
                step()  # model's turn

    if st.session_state.state == ActionState.EVALUATION_WAITING:
        with st.spinner("Evaluating..."):
            step()

    with st.expander("Background", expanded=False):
        agent_infos = compose_agents()
        for agent_idx, agent_info in enumerate(agent_infos):
            st.markdown(
                f"""
                <details>
                    <summary>Agent {agent_idx + 1} Background</summary>
                    {agent_info}
                </details>
            """,
                unsafe_allow_html=True,
            )

    with st.expander("Chat History", expanded=True):
        pretty_print_messages(compose_messages()[:-3])

    with st.expander("Evaluation"):
        pretty_print_evaluation(compose_messages()[-2:])


from sotopia.database import EpisodeLog


def pretty_print_evaluation(messages: list[str]) -> None:
    assert len(messages) == 2, "Evaluation messages should be of length 2"
    st.markdown(
        f"""
            <details>
                <summary> Evaluation </summary>
                Reasoning: {messages[0]}
            </details>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
         Rewards: {messages[1]}
        """,
        unsafe_allow_html=True,
    )


def pretty_print_messages(messages: list[str]) -> None:
    for index, message in enumerate(messages):
        message = message.replace("\n", "<br />")
        st.markdown(
            f"""
            <details>
                <summary>Message {index + 1}</summary>
                {message}
            </details>
            """,
            unsafe_allow_html=True,
        )


def compose_agents() -> list[str]:  # type: ignore
    agents = st.session_state.agents
    from sotopia.envs.parallel import _agent_profile_to_friendabove_self

    agent_to_render = [
        _agent_profile_to_friendabove_self(agent.profile, agent_id)
        for agent_id, agent in enumerate(agents.values())
    ]
    return agent_to_render


def compose_messages() -> list[str]:  # type: ignore
    env = st.session_state.env
    agent_list = list(st.session_state.agents.values())

    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag="tmp",
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in st.session_state.messages
        ],
        reasoning=st.session_state.reasoning,
        rewards=st.session_state.rewards,
        rewards_prompt="",
    )
    ep_to_render = epilog.render_for_humans()[1]
    ep_to_render[0] = ep_to_render[0].split("Conversation Starts:")[-1].strip()

    return ep_to_render


def set_settings(
    agent_choice_1, agent_choice_2, scenario_choice, user_position
) -> None:  # type: ignore
    global HUMAN_AGENT_IDX
    scenarios = st.session_state.env_mapping
    agent_list_1, agent_list_2 = st.session_state.agent_mapping

    env_agent_combo = EnvAgentComboStorage(
        env_id=scenarios[scenario_choice].pk,
        agent_ids=[agent_list_1[agent_choice_1].pk, agent_list_2[agent_choice_2].pk],
    )
    HUMAN_AGENT_IDX = 0 if user_position == "First Agent" else 1
    env, agents, environment_messages = get_env_agents(env_agent_combo)

    st.session_state.env = env
    st.session_state.agents = agents
    st.session_state.environment_messages = environment_messages
    st.session_state.messages = (
        [
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        ]
        if st.session_state.messages == []
        else st.session_state.messages
    )
