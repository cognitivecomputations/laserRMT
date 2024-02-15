from lib.utils.assets import PromptTemplate


def get_llm_prompt(
    system_insturction,
    user_prompt="",
    prompt_format: PromptTemplate = PromptTemplate.chatml,
    sep=" ",
):
    if prompt_format == PromptTemplate.chatml:
        return f"""<|im_start|>system
{system_insturction}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

    elif prompt_format == PromptTemplate.instruct:
        if user_prompt != "":
            system_insturction = system_insturction + sep + user_prompt

        return f"[INST] {system_insturction} [/INST]"

    else:
        raise Exception('prompt_format must be "chatml" or "mixtral"')
