import asyncio
import json
from functools import wraps
from typing import cast

import streamlit as st
from sotopia.agents import Agents, LLMAgent
from sotopia.database import (
    AgentProfile,
    EnvAgentComboStorage,
    EnvironmentProfile,
    EpisodeLog,
)
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
)
from sotopia.envs.parallel import (
    _agent_profile_to_friendabove_self,
    render_text_for_agent,
)
from sotopia.messages import AgentAction

from socialstream.utils import messageForRendering, render_for_humans


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
        st.session_state.env_agent_combo = EnvAgentComboStorage.find().all()[0]
        st.session_state.state = ActionState.IDLE
        st.session_state.env = None
        st.session_state.agents = None
        st.session_state.environment_messages = None
        st.session_state.messages = []
        st.session_state.rewards = [0.0, 0.0]
        st.session_state.reasoning = ""

    all_agents = AgentProfile.find().all()[:10]
    all_envs = EnvironmentProfile.find().all()[:10]

    st.session_state.agent_mapping = [
        {get_full_name(agent_profile): agent_profile for agent_profile in all_agents}
    ] * 2
    st.session_state.env_mapping = {
        env_profile.codename: env_profile for env_profile in all_envs
    }


def step(user_input: str | None = None) -> None:
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


MODEL_LIST = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "together_ai/meta-llama/Llama-3-8b-chat-hf",
    "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
]


def chat_demo() -> None:
    initialize_session_state()

    with st.expander("Create your scenario!", expanded=True):
        scenarios = st.session_state.env_mapping
        agent_list_1, agent_list_2 = st.session_state.agent_mapping

        scenario_col, model_col = st.columns(2)
        with scenario_col:
            scenario_choice = st.selectbox(
                "Choose a scenario:",
                scenarios.keys(),
                disabled=st.session_state.active,
                index=0,
            )
        with model_col:
            model_choice = st.selectbox(
                "Choose a model:", MODEL_LIST, disabled=st.session_state.active, index=0
            )

        agent_col1, agent_col2 = st.columns(2)
        with agent_col1:
            agent_choice_1 = st.selectbox(
                "Choose the first agent:",
                agent_list_1.keys(),
                disabled=st.session_state.active,
                index=0,
            )
        with agent_col2:
            agent_choice_2 = st.selectbox(
                "Choose the second agent:",
                agent_list_2.keys(),
                disabled=st.session_state.active,
                index=1,
            )
        user_position = st.selectbox(
            "Which agent do you want to be?",
            [agent_choice_1, agent_choice_2],
            disabled=st.session_state.active,
        )
        # user_position = st.selectbox(
        #     "Do you want to be the first or second agent?",
        #     POSITION_CHOICES,
        #     disabled=st.session_state.active,
        # )
        if agent_choice_1 == agent_choice_2:
            st.warning(
                "The two agents cannot be the same. Please select different agents."
            )
            st.stop()

        if (
            agent_choice_1
            or agent_choice_2
            or user_position
            or scenario_choice
            or model_choice
        ) and not st.session_state.active:
            global MODEL
            MODEL = model_choice
            print(f"Setting settings with model {MODEL}...")
            set_settings(
                agent_choice_1,
                agent_choice_2,
                scenario_choice,
                user_position,
                [agent_choice_1, agent_choice_2],
            )

    with st.expander("Check your social task!", expanded=True):
        agent_infos = compose_agent_messages()
        env_info, goals_info = compose_env_messages()

        st.markdown(
            f"""**Scenario:** {env_info}""",
            unsafe_allow_html=True,
        )

        agent1_col, agent2_col = st.columns(2)
        agent_cols = [agent1_col, agent2_col]
        for agent_idx, agent_info in enumerate(agent_infos):
            agent_col = agent_cols[agent_idx]
            with agent_col:
                st.markdown(
                    f"""**Agent {agent_idx + 1} Background:** {agent_info}""",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""**Now you are Agent {HUMAN_AGENT_IDX + 1}.** Your goal: {goals_info[HUMAN_AGENT_IDX]}"""
        )

    start_col, stop_col = st.columns(2)
    with start_col:
        start_button = st.button("Start", disabled=st.session_state.active)
        if start_button:
            set_settings(
                agent_choice_1,
                agent_choice_2,
                scenario_choice,
                user_position,
                agent_names=[agent_choice_1, agent_choice_2],
                reset_msgs=True,
            )
            st.session_state.active = True
            st.session_state.state = (
                ActionState.HUMAN_WAITING
                if user_position == "First Agent"
                else ActionState.MODEL_WAITING
            )

            if st.session_state.state == ActionState.MODEL_WAITING:
                with st.spinner("Model acting..."):
                    step()  # model's turn

    with stop_col:
        stop_button = st.button("Stop", disabled=not st.session_state.active)
        if stop_button and st.session_state.active:
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
        print("Evaluating...")
        with st.spinner("Evaluating..."):
            step()
            st.rerun()

    messages = render_messages()
    tag_for_eval = ["Agent 1", "Agent 2", "General"]
    chat_history = [
        message for message in messages if message["role"] not in tag_for_eval
    ]
    evaluation = [message for message in messages if message["role"] in tag_for_eval]

    with st.expander("Chat History", expanded=True):
        streamlit_rendering(chat_history)

    with st.expander("Evaluation"):
        # a small bug: when there is a agent not saying anything there will be no separate evaluation for that agent
        streamlit_rendering(evaluation)


def streamlit_rendering(messages: list[messageForRendering]) -> None:
    agent1_name, agent2_name = list(st.session_state.agents.keys())[:2]
    agent_color_mapping = {
        agent1_name: "lightblue",
        agent2_name: "green",
    }

    avatar_mapping = {
        "env": "🌍",
        "obs": "🌍",
    }
    if HUMAN_AGENT_IDX == 0:
        avatar_mapping[agent1_name] = "👤"
        avatar_mapping[agent2_name] = "🤖"
    else:
        avatar_mapping[agent1_name] = "🤖"
        avatar_mapping[agent2_name] = "👤"

    role_mapping = {
        "Background Info": "background",
        "System": "info",
        "Environment": "env",
        "Observation": "obs",
        "General": "info",
        "Agent 1": "info",
        "Agent 2": "info",
        agent1_name: agent1_name,
        agent2_name: agent2_name,
    }

    for index, message in enumerate(messages):
        role = role_mapping.get(message["role"], "info")
        content = message["content"]

        if role == "background":
            continue

        if role == "obs" or message.get("type") == "action":
            content = json.loads(content)

        with st.chat_message(role, avatar=avatar_mapping.get(role, None)):
            if isinstance(content, dict):
                st.json(content)
            elif role == "info":
                st.markdown(
                    f"""
                    <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif index < 2:  # Apply foldable for the first two messages
                continue
            else:
                st.markdown(content.replace("\n", "<br />"), unsafe_allow_html=True)


def compose_env_messages() -> tuple[str, list[str]]:
    env: ParallelSotopiaEnv = st.session_state.env
    env_profile = EnvironmentProfile.get(env.profile.pk)
    env_to_render = env_profile.scenario
    goals_to_render = env_profile.agent_goals

    return env_to_render, goals_to_render


def compose_agent_messages() -> list[str]:  # type: ignore
    agents = st.session_state.agents

    agent_to_render = [
        render_text_for_agent(
            raw_text=_agent_profile_to_friendabove_self(agent.profile, agent_id),
            agent_id=HUMAN_AGENT_IDX,
        )
        for agent_id, agent in enumerate(agents.values())
    ]
    print(agent_to_render)
    return agent_to_render


def render_messages() -> list[messageForRendering]:
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
    rendered_messages = render_for_humans(epilog)
    return rendered_messages


def set_settings(
    agent_choice_1: str,
    agent_choice_2: str,
    scenario_choice: str,
    user_agent_name: str,
    agent_names: list[str],
    reset_msgs: bool = False,
) -> None:  # type: ignore
    global HUMAN_AGENT_IDX
    scenarios = st.session_state.env_mapping
    agent_map_1, agent_map_2 = st.session_state.agent_mapping

    env_agent_combo = EnvAgentComboStorage(
        env_id=scenarios[scenario_choice].pk,
        agent_ids=[agent_map_1[agent_choice_1].pk, agent_map_2[agent_choice_2].pk],
    )
    for agent_name in agent_names:
        if agent_name == user_agent_name:
            HUMAN_AGENT_IDX = agent_names.index(agent_name)
            break
    # HUMAN_AGENT_IDX = 0 if user_position == "First Agent" else 1
    # TODO decide whether to use agent name of the index
    env, agents, environment_messages = get_env_agents(env_agent_combo)

    st.session_state.env = env
    st.session_state.agents = agents
    st.session_state.environment_messages = environment_messages
    if reset_msgs:
        st.session_state.messages = []
        st.session_state.reasoning = ""
        st.session_state.rewards = [0.0, 0.0]
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
