import asyncio
from functools import wraps
from typing import Optional, TypedDict, cast

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
    render_text_for_environment,
)
from sotopia.messages import AgentAction, Observation

HUMAN_MODEL_NAME = "human"
MODEL_LIST = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "together_ai/meta-llama/Llama-3-8b-chat-hf",
    "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    HUMAN_MODEL_NAME,
]


class ActionState_v0:
    IDLE = 1
    HUMAN_WAITING = 2
    HUMAN_SPEAKING = 3
    MODEL_WAITING = 4
    MODEL_SPEAKING = 4
    EVALUATION_WAITING = 5


class ActionState:
    IDLE = 1
    AGENT1_WAITING = 2
    AGENT1_SPEAKING = 3
    AGENT2_WAITING = 4
    AGENT2_SPEAKING = 5
    EVALUATION_WAITING = 6


WAIT_STATE: list[int] = [ActionState.AGENT1_WAITING, ActionState.AGENT2_WAITING]
SPEAK_STATE: list[int] = [ActionState.AGENT1_SPEAKING, ActionState.AGENT2_SPEAKING]


def get_full_name(agent_profile: AgentProfile) -> str:
    return f"{agent_profile.first_name} {agent_profile.last_name}"


def print_current_speaker() -> None:
    match st.session_state.state:
        case ActionState.AGENT1_SPEAKING:
            print("Agent 1 is speaking...")
        case ActionState.AGENT2_SPEAKING:
            print("Agent 2 is speaking...")
        case ActionState.AGENT1_WAITING:
            print("Agent 1 is waiting...")
        case ActionState.AGENT2_WAITING:
            print("Agent 2 is waiting...")
        case ActionState.EVALUATION_WAITING:
            print("Evaluation is waiting...")


class EnvAgentProfileCombo:
    def __init__(self, env: EnvironmentProfile, agents: list[AgentProfile]) -> None:
        self.env = env
        self.agents = agents


def async_to_sync(async_func: callable) -> callable:
    @wraps(async_func)
    def sync_func(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_func(*args, **kwargs))
        loop.close()
        return result

    return sync_func


import json


def get_abstract(description: str) -> str:
    return " ".join(description.split()[:50]) + "..."


def load_additional_agents() -> list[AgentProfile]:
    data_file = "data/negotiation_agents.json"
    agents = [AgentProfile(**agent_data) for agent_data in json.load(open(data_file))]
    return agents


def load_additional_envs() -> list[EnvironmentProfile]:
    data_file = "data/negotiation_scenarios.json"
    envs = [EnvironmentProfile(**env_data) for env_data in json.load(open(data_file))]
    return envs


def initialize_session_state(force_reload: bool = False) -> None:
    all_agents = AgentProfile.find().all()[:10]
    all_envs = EnvironmentProfile.find().all()[:10]
    additional_agents = load_additional_agents()
    additional_envs = load_additional_envs()

    all_agents = additional_agents + all_agents
    all_envs = additional_envs + all_envs

    st.session_state.agent_mapping = [
        {get_full_name(agent_profile): agent_profile for agent_profile in all_agents}
    ] * 2
    st.session_state.env_mapping = {
        env_profile.codename: env_profile for env_profile in all_envs
    }
    st.session_state.env_description_mapping = {
        env_profile.codename: get_abstract(
            render_text_for_environment(env_profile.scenario)
        )
        for env_profile in all_envs
    }

    if "active" not in st.session_state or force_reload:
        st.session_state.active = False
        st.session_state.conversation = []
        st.session_state.background = "Default Background"
        st.session_state.env_agent_combo = EnvAgentComboStorage.find().all()[0]
        st.session_state.state = ActionState.IDLE
        st.session_state.env = None
        st.session_state.agents = None
        st.session_state.environment_messages = None
        st.session_state.messages = []
        st.session_state.agent_models = ["gpt-4o-mini", "gpt-4o-mini"]
        st.session_state.evaluator_model = "gpt-4o"
        st.session_state.editable = False
        st.session_state.human_agent_idx = 0

        st.session_state.rewards = [0.0, 0.0]
        st.session_state.reasoning = ""
        set_settings(
            agent_choice_1=get_full_name(all_agents[0]),
            agent_choice_2=get_full_name(all_agents[1]),
            scenario_choice=all_envs[0].codename,
            user_agent_name="PLACEHOLDER",
            agent_names=[],
            reset_agents=True,
        )


def set_from_env_agent_profile_combo(
    env_agent_combo: EnvAgentProfileCombo, reset_msgs: bool = False
) -> None:
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


def get_env_agents(
    env_agent_combo: EnvAgentProfileCombo,
) -> tuple[ParallelSotopiaEnv, Agents, dict[str, Observation]]:
    environment_profile = env_agent_combo.env
    agent_profiles = env_agent_combo.agents
    agent_list = [
        LLMAgent(
            agent_profile=agent_profile,
            model_name=st.session_state.agent_models[agent_idx],
        )
        for agent_idx, agent_profile in enumerate(agent_profiles)
    ]
    for idx, goal in enumerate(environment_profile.agent_goals):
        agent_list[idx].goal = goal

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env = ParallelSotopiaEnv(
        action_order="round-robin",
        model_name=st.session_state.evaluator_model,
        evaluators=[
            RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
        ],
        terminal_evaluators=[
            ReachGoalLLMEvaluator(
                st.session_state.evaluator_model,
                EvaluationForTwoAgents[SotopiaDimensions],
            ),
        ],
        env_profile=environment_profile,
    )

    environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()

    return env, agents, environment_messages


def set_settings(
    agent_choice_1: str,
    agent_choice_2: str,
    scenario_choice: str,
    user_agent_name: str,
    agent_names: list[str],
    reset_msgs: bool = False,
    reset_agents: bool = True,
) -> None:  # type: ignore
    scenarios = st.session_state.env_mapping
    agent_map_1, agent_map_2 = st.session_state.agent_mapping

    env = scenarios[scenario_choice] if reset_agents else st.session_state.env.profile
    agents = (
        [agent_map_1[agent_choice_1], agent_map_2[agent_choice_2]]
        if reset_agents
        else [agent.profile for agent in st.session_state.agents.values()]
    )

    env_agent_combo = EnvAgentProfileCombo(
        env=env,
        agents=agents,
    )
    set_from_env_agent_profile_combo(
        env_agent_combo=env_agent_combo, reset_msgs=reset_msgs
    )


def step(user_input: str | None = None) -> None:
    print_current_speaker()
    env: ParallelSotopiaEnv = st.session_state.env
    print("Env profile: ", env.profile)
    for agent_name in env.agents:
        print("Agent profile: ", st.session_state.agents[agent_name].profile)
        print("Agent goal: ", st.session_state.agents[agent_name].goal)

    def act_function(
        user_input: Optional[str], is_human: bool, agent_name: str
    ) -> AgentAction:
        assert user_input is not None or not is_human, "User input is required"
        if is_human:
            st.session_state.agents[agent_name].recv_message(
                "Environment", st.session_state.environment_messages[agent_name]
            )
            # set the message to the agents
            return AgentAction(action_type="speak", argument=user_input)
        else:
            return async_to_sync(st.session_state.agents[agent_name].aact)(
                st.session_state.environment_messages[agent_name]
            )  # type: ignore

    agent_messages: dict[str, AgentAction] = dict()
    actions = []
    # AGENT1_WAITING -> AGENT1_SPEAKING
    for agent_idx, agent_name in enumerate(env.agents):
        session_state: ActionState = st.session_state.state
        model_in_turn = st.session_state.agent_models[agent_idx]
        is_human = model_in_turn == HUMAN_MODEL_NAME

        agent_speaking = (
            agent_idx == 0 and session_state == ActionState.AGENT1_SPEAKING
        ) or (agent_idx == 1 and session_state == ActionState.AGENT2_SPEAKING)
        if not agent_speaking:
            st.session_state.agents[agent_name].recv_message(
                "Environment", st.session_state.environment_messages[agent_name]
            )

        match agent_idx:
            case 0:
                match session_state:
                    case ActionState.AGENT1_SPEAKING:
                        action = act_function(
                            user_input, is_human=is_human, agent_name=agent_name
                        )
                    case ActionState.EVALUATION_WAITING:
                        action = AgentAction(action_type="leave", argument="")
                    case _:
                        action = AgentAction(action_type="none", argument="")

            case 1:
                match session_state:
                    case ActionState.AGENT2_SPEAKING:
                        action = act_function(
                            user_input, is_human=is_human, agent_name=agent_name
                        )
                    case ActionState.EVALUATION_WAITING:
                        action = AgentAction(action_type="leave", argument="")
                    case _:
                        action = AgentAction(action_type="none", argument="")
        print(f"Agent {agent_idx} model {model_in_turn} output action: ", action)

        actions.append(action)

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

    session_state: ActionState = st.session_state.state
    match session_state:
        case ActionState.AGENT1_SPEAKING:
            st.session_state.state = ActionState.AGENT2_WAITING
        case ActionState.AGENT2_SPEAKING:
            st.session_state.state = ActionState.AGENT1_WAITING
        case ActionState.EVALUATION_WAITING:
            st.session_state.state = ActionState.IDLE
        case ActionState.IDLE:
            st.session_state.state = ActionState.IDLE
        case _:
            raise ValueError("Invalid state", st.session_state.state)

    done = all(terminated.values())


# def _agent_profile_to_friendabove_self(profile: AgentProfile, agent_id: int) -> str:
#     return f"{profile.first_name} {profile.last_name} is a {profile.age}-year-old {_map_gender_to_adj(profile.gender)} {profile.occupation.lower()}. {profile.gender_pronoun} pronouns. {profile.public_info} Personality and values description: {profile.personality_and_values} <p viewer='agent_{agent_id}'>{profile.first_name}'s secrets: {profile.secret}. Big five type: {profile.big_five}</p>"
