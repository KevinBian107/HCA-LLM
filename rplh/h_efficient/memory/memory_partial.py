import sys
from pathlib import Path
import argparse

main_path = Path(__file__).resolve().parent.parent.parent
if str(main_path) not in sys.path:
    sys.path.append(str(main_path))
    
from rplh.llm.language_model import *
import tiktoken
from rplh.h_efficient.memory.memory import *

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
N = 5

def rplh_prompt_partial_func(
    state_update_prompt: str,
    data: dict,
    dialogue_history_method: str,
    HCA_agent_location: str,
    feedback: str = "",
) -> str:
    """
    Designs an input prompt for a role-playing leader-hallucinating (RPLH) agent
    using in-context learning and chain-of-thought reasoning.

    Args:
        state_update_prompt (str): Description of the current state and available actions.
        data (dict): Dictionary containing past responses, states, and dialogue history.
        dialogue_history_method (str): Method to handle dialogue history, e.g.,
                                       "_w_only_state_action_history", "_w_compressed_dialogue_history".
        HCA_agent_location (str): Location of the HCA agent in the grid.

    Returns:
        str: A structured prompt for the role-playing leader-hallucinating agent.

    Notes:
        Boxes just need to be moved to the target location, not in the target location.
    """

    if data["env_step"] == 0:
        attitude = None
        success_action = f"""No previous action, here is an sample where box_x and box_y are arbitrary boxes:
        {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_y, target_y])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        success_action = data["response_total_list"][-1]

    response_total_list = data["response_total_list"]
    pg_state_list = data["pg_state_list"]
    dialogue_history_list = data["dialogue_history_list"]

    print(f"HISTORY METHOD: {dialogue_history_method}")

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")

    user_prompt_1 = f"""..."""  # check for prompt length, no need for us
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
        "_w_markovian_state_action_history",
        "_w_no_history",
    ):

        # first iteration no summary
        if dialogue_history_method == "_w_markovian_state_action_history":
            state_action_prompt = ""
            # Markovian state-action history
            previous_state_idx = len(response_total_list) - 1
            if previous_state_idx != -1:
                state_action_prompt = f"""
            Previous State: {better_state_repres(pg_state_list[previous_state_idx])}
            Previous Action: {response_total_list[previous_state_idx]}\n\n
            """
        elif dialogue_history_method == "_w_no_history":
            state_action_prompt = ""
        elif dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_all_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        if attitude == None:
            print("ATTITUDE IS NONE")
            att_promt = ""
        else:
            att_promt = f"""
            Please learn from attitude in the following ways:

                1. Please undrstand the attitude of each agents in this environment,
                including yourself based on this attitude report given from another agent: 
                
                {attitude}.

                2. Based on this charcteristics of each agent, please do two things and added them after each agent's attitude:
                    i. Reason about the reactions each agent would have towards your command.
                    ii. Reason about how they would give actions if they are the central agent.
            
            Use the following format:
            - Attitude of agent...
            - Reaction of agent...
            """
        if feedback != "":
            feedback = (
                "There is error in preivous action plan. Here is the feedbcak: "
                + feedback
            )
        
        import re
        # Regular expression to extract Agent[0.5, 0.5] data
        escaped_agent = re.escape(HCA_agent_location)
        pattern = fr"{escaped_agent}:.*?(?=Agent\[|$)"
        print(pattern)
        match = re.search(pattern, state_update_prompt, re.DOTALL)
        agent_data = match.group(0) if match else None
        print(f'AGENT CAN SEE AND DO: {agent_data}')

        HCA_prompt = f"""
            You are a central planner directing agent in a grid-like field to move colored boxes.
            You are agent at grid [{HCA_agent_location}]. You need to make moves and other agents need to make moves as well.
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Your task is to instruct each agent to match all boxes to their color-coded targets.
            After each move, agents provide updates for the next sequence of actions.
            You are the central agent and your job is to coordinate the agents optimally.
            
            The previous state and action pairs at each step are: {state_action_prompt}
            Based on this previous state action interaction, you need to reason what each agent can and cannot do and output in world_model.
            
            Notice that you don't see all the environmental state, you can only observe what you can see and make your plan based on your beliefs of what the environmental looks like and  make action plan based on such beliefs.
            Other agent in the environment may observe different things than you do and may make comments to your plan later.
            Try to make conservative actions as you don't know everything
            
            Please imagine what each agent can and caannot do first based on the goals mentioned ealier. Remanber to reason for all agent, inclduing yourself.
            
            The possible that you can take is: {agent_data}.
            
            Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.

            {att_promt}

            Think about what the future {N} actions would be if you want to achieve the goal with the reasoning.
            Remanber to wirte out for each step, what you plan for every agent to do and what would the state change be.
            
            {feedback}

            Action Plan:
            Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])","Agent[0.5, 1.5]": "move(box_y, target_y)"}} where box_x and box_y are arbitrary boxes.
            Try to propose actions for all four agents.
            One agent can only make one action.
            No agent name should be given if the agent does not have a task next. 
            """
    return HCA_prompt


def dialogue_partial_func(
    state_update_prompt_local_agent: str,
    state_update_prompt_other_agent: str,
    central_response: str,
    data: dict,
    dialogue_history_method: str,
    local_agent_location: str,
    feedback: str = "",
) -> str:
    """
    Constructs a dialogue prompt for a local agent in response to the central planner.

    Args:
        state_update_prompt_local_agent (str): State and actions specific to the local agent.
        state_update_prompt_other_agent (str): State and actions of other agents.
        central_response (str): Central planner's response.
        data (dict): Data containing historical responses and states.
        dialogue_history_method (str): Method for managing dialogue history.
        local_agent_location (str): Location of the local agent in the grid.

    Returns:
        str: Dialogue prompt for the local agent. 
    """

    if data["env_step"] == 0:
        attitude = None
        success_action = f"""No previous action, here is an sample where box_x and box_y are arbitrary boxes:
        {{"Agent[0.5, 0.5]":"move(box_x, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_y, target_y])"}}"""
    else:
        attitude = data["attitude_info"][-1]
        success_action = data["response_total_list"][-1]

    if attitude == None:
        print("ATTITUDE IS NONE")
        att_promt = "Be very critical"
    else:
        att_promt = f"""
            Please pick up an attitude on this problem for yourself based on the attitude that this attitude report assigned to you: {attitude}.
        """

    response_total_list = data["response_total_list"]
    pg_state_list = data["pg_state_list"]
    dialogue_history_list = data["dialogue_history_list"]

    if len(pg_state_list) - len(response_total_list) != 1:
        raise ValueError("state and response list do not match")
    # if len(pg_state_list) - len(dialogue_history_list) != 1:
    #     raise ValueError("state and dialogue history list do not match")

    user_prompt_1 = f"""..."""  # check for prompt length, no need for us
    token_num_count = len(enc.encode(user_prompt_1))

    if dialogue_history_method in (
        "_w_only_state_action_history",
        "_w_compressed_dialogue_history",
        "_w_all_dialogue_history",
        "_w_markovian_state_action_history",
        "_w_no_history"
    ):
        # first iteration no summary
        if dialogue_history_method == "_w_markovian_state_action_history":
            state_action_prompt = ""
            # Markovian state-action history
            previous_state_idx = len(response_total_list) - 1
            if previous_state_idx != -1:
                state_action_prompt = f"""
            Previous State: {better_state_repres(pg_state_list[previous_state_idx])}
            Previous Action: {response_total_list[previous_state_idx]}\n\n
            """
        elif dialogue_history_method == "_w_no_history":
            state_action_prompt = ""
        elif dialogue_history_method == "_w_only_state_action_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_compressed_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break
        elif dialogue_history_method == "_w_all_dialogue_history":
            state_action_prompt = ""
            for i in range(len(response_total_list) - 1, -1, -1):
                state_action_prompt_next = (
                    f"State{i + 1}: {better_state_repres(pg_state_list[i])}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n"
                    + state_action_prompt
                )
                if (
                    token_num_count + len(enc.encode(state_action_prompt_next))
                    < input_prompt_token_limit
                ):
                    state_action_prompt = state_action_prompt_next
                else:
                    break

        local_HCA_prompt = f"""
            Imagine that you are a central planner directing agent in a grid-like field to move colored boxes.
            Particularly, you're a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground at grid location of [{local_agent_location}].
            
            The goals and rules of this environment are:
            {GOAL_RULES}
            
            Other central planner is also coordinating all other agents to achieve the goal: match each box with its color-coded target.
            You can only move same color boxes to same color targets.
            
            Notice that you don't see all the environmental state, you can only observe what you can see and make your plan based on your beliefs of what the environmental looks like and  make action plan based on such beliefs.
            Other agent in the environment may observe different things than you do and may make comments to your plan later.
            You need to help modify this plan by including your perspectives.
            Try to add more actions instead of deleting actions from the action plan.
            
            The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
            
            Please only plan actions for each agent that is chosen from each agent's doable action list, do not give a action that is not doable.
            
            The previous state and action pairs at each step are: {state_action_prompt}
            Please learn from previous steps in a few steps:
                
                1. {att_promt}

                2. You would recieve an plan from the other central planner, please evaluate the given plan and give critical feedbacks.

            You can imagine first about how you would plan these actions and specify your action plan.
            This is the success response of previous state: {success_action}
            Remanber to assign action to your self as well.

            The other central planner's current action plan is giving as: {central_response}.
            Try to find agreement with the central ageent if you can, the goal is to resolve conversation.
            
            Prioritize adding more actions or keeping at least the same number of action if possible, but the number of action should not be more than the number of agents.

            Please evaluate the given plan.
            If you agree with it, respond 'I Agree', without any extra words.
            If not, briefly explain your objections to this other central planner and an judger agent will get involved.
            Ensure that you still include the actions that you agree with.
            
            Please remanber to only change the actions you disagree with and not change other actions, remanber to include all actions for each agents.
            
            {feedback}
            
            Your response:
        """
    return local_HCA_prompt